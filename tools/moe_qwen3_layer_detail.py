#!/usr/bin/env python3
"""Decode saved layer-2 replays and audit card/core dependencies; no device execution."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from moe_qwen3_profile import canon


STEM = '/model/layers.2/mlp/'
HOT = {'MatMul', 'MatMul_1', 'MatMul_2'}
COLD = {'MatMul_3', 'MatMul_4', 'MatMul_5'}
CORE = re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$')
PHASES = ['initial_pack_ms', 'group0_ms', 'between_groups_ms', 'group1_ms',
          'last_unpack_ms', 'combine_ms', 'tail_ms']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'Empty artifact: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema_from_sdk(binary):
    """Use the installed decoder's own protobuf schema, not inferred wire fields."""
    data = binary.read_bytes()
    start = data.index(b'\n\x14AICOpstatsDesc.proto')

    def varint(pos):
        value, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7

    pos = start
    while data[pos]:
        tag, nxt = varint(pos)
        if tag & 7 != 2:
            break
        size, nxt = varint(nxt)
        pos = nxt + size
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    raw = data[start:pos]
    desc = descriptor_pb2.FileDescriptorProto()
    desc.ParseFromString(raw)
    if desc.name != 'AICOpstatsDesc.proto':
        raise ValueError('Unexpected SDK descriptor schema')
    pool = descriptor_pool.DescriptorPool()
    pool.Add(desc)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName('aicopstatsdesc.OpstatsDescriptor'))
    return raw, cls


def metadata(args, capacity, cls):
    target = args.out / f'c{capacity}' / 'metadata'
    if not target.exists():
        subprocess.run([str(args.sdk/'tools/qaic-qpc'), 'extract', '--qpc',
                        str(args.scratch/f'c{capacity}_compile/qpc/programqpc.bin'),
                        '--output-dir', str(target), '-s', '*opstatsdesc.bin'],
                       check=True, stdout=subprocess.DEVNULL)
    paths = sorted(target.glob('QAicGraph_slice*_dir/opstatsdesc.bin'))
    if len(paths) != 4:
        raise ValueError('Need four compiler descriptors')
    lookup, hashes = {}, {}
    for path in paths:
        card = int(re.search(r'slice(\d+)', str(path))[1])
        obj = cls()
        obj.ParseFromString(path.read_bytes())
        if (obj.major_version, obj.minor_version) != (1, 11):
            raise ValueError('Recheck descriptor assumptions for this SDK version')
        common = obj.opstats_metadata.common_metadata.op_details
        kinds = {p.id: p.str.strip() for p in common.op_kind_id_str_pairs}
        memories = {p.id: p.str for p in common.op_memory_id_str_pairs}
        for spec in obj.opstats_metadata.specialization_metadata:
            if spec.name != f'QAicGraph_slice{card:02}':
                raise ValueError('Unexpected specialization')
            for core, groups in enumerate(spec.core_op_name_kinds):
                for group in groups.tg_op_name_kinds:
                    for op in group.op_name_kinds:
                        key = (card, core, int(op.oc))
                        if key in lookup:
                            raise ValueError('Ambiguous compiler operation ID')
                        lookup[key] = dict(name=op.name, bytes=int(op.output_size),
                                           kind=kinds[op.kind_id], memory=memories[op.memory_id])
        hashes[str(path)] = sha(path)
    write(target/'hashes.json', hashes)
    return lookup


def event_key(event):
    return event['pid'], event['tid'], int(event['args']['oc'])


def analyze(path, capacity, sample, lookup, out):
    raw = read(path)['traceEvents']
    threads = {(e['pid'], e['tid']): e['args']['name'] for e in raw
               if e['ph'] == 'M' and e['name'] == 'thread_name'}
    work, waits, execution = {}, [], []
    for event in raw:
        if event['ph'] != 'X':
            continue
        args = event.get('args', {})
        thread = threads.get((event['pid'], event['tid']), '')
        match = CORE.search(thread)
        card, core, engine = (int(match[1]), int(match[2]), match[3] or 'execution') if match else (-1, -1, 'P2P')
        start, dur = float(event['ts']), float(event.get('dur', 0))
        if engine == 'execution':
            execution.append((start, start+dur))
            continue
        if 'oc' not in args:
            continue
        is_wait = event['name'].startswith('sync ')
        if event['name'].startswith('barrier '):
            continue
        if is_wait:
            dur = float(args.get('opSyncDurUs', dur))
        meta = lookup.get((card, core, int(args['oc'])))
        if match and not meta:
            raise ValueError(f'No compiler metadata for {event_key(event)}')
        if meta and (meta['kind'] != args.get('opKind', '').strip()
                     or canon(meta['name']) != canon(args.get('opName', ''))):
            raise ValueError('Trace and compiler metadata disagree')
        row = dict(card=card, core=core, engine=engine, kind=args.get('opKind', '').strip(),
                   memory=args.get('opMemory', ''), node=canon(args.get('opName', '')),
                   lowered_op=args.get('opName', ''), oc=int(args['oc']),
                   start_us=start, end_us=start+dur, duration_us=dur,
                   bytes=int(args.get('opOutputSize', meta['bytes'] if meta else 0)),
                   port=args.get('PortDescription', ''), transaction=args.get('TransactionId', ''),
                   pid=event['pid'], tid=event['tid'])
        if is_wait:
            waits.append(row)
        else:
            key = event_key(event)
            if key in work:
                raise ValueError('Ambiguous trace operation ID within thread')
            work[key] = row
    origin, finish = min(a for a, _ in execution), max(b for _, b in execution)
    hmx = [r for r in work.values() if r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
    bounds, core_rows = {}, []
    for stage, names in [('hot', HOT), ('cold', COLD)]:
        selected = [r for r in hmx if r['node'].removeprefix(STEM) in names]
        lo, hi = min(r['start_us'] for r in selected), max(r['end_us'] for r in selected)
        bounds[stage] = lo, hi
        for card in range(4):
            for core in range(16):
                rows = [r for r in selected if (r['card'], r['core']) == (card, core)]
                if not rows:
                    raise ValueError('Missing HMX core')
                sw = [r for r in waits if r['engine'] == 'HMX' and
                      r['node'].removeprefix(STEM) in names and (r['card'], r['core']) == (card, core)]
                if len(sw) != len(rows):
                    raise ValueError('Unpaired HMX waits')
                sw.sort(key=lambda r: r['start_us'])
                if any(a['end_us'] > b['start_us']+.01 for a, b in zip(sw, sw[1:])):
                    raise ValueError('Overlapping HMX waits')
                errors = [abs(w['end_us']-work[w['pid'], w['tid'], w['oc']]['start_us']) for w in sw]
                if max(errors) > 1:
                    raise ValueError('Wait end does not match compute start')
                down = [r for r in rows if r['node'] == STEM+('MatMul_2' if stage == 'hot' else 'MatMul_5')]
                core_rows.append(dict(stage=stage, card=card, core=core, events=len(rows),
                    start_ms=min(r['start_us'] for r in rows)/1000,
                    end_ms=max(r['end_us'] for r in rows)/1000,
                    first_down_ms=min(r['start_us'] for r in down)/1000,
                    compute_ms=sum(r['duration_us'] for r in rows)/1000,
                    wait_ms=sum(max(0, min(hi, r['end_us'])-max(lo, r['start_us'])) for r in sw)/1000,
                    wait_pair_error_us=max(errors)))
    local = [r for r in work.values() if r['node'] == STEM+'Einsum_3' and
             r['engine'] == 'HVX' and r['kind'] == 'aicbatchedreduceadd']
    final = [r for r in work.values() if r['node'] == STEM+'Einsum_4' and
             r['engine'] in ['HVX', 'HMX'] and r['kind'] != 'aicendcyclestats']
    edges = [origin, *bounds['hot'], *bounds['cold'], min(r['start_us'] for r in local),
             max(r['end_us'] for r in final), finish]
    if edges != sorted(edges):
        raise ValueError('Nonadditive phase boundaries')
    phases = {name: (b-a)/1000 for name, a, b in zip(PHASES, edges, edges[1:])}
    assert abs(sum(phases.values())-(finish-origin)/1000) < 1e-9

    # The SDK reuses flow IDs on different cards: pair in file order, not by a global ID join.
    pending, flows, incoming = {}, [], defaultdict(list)
    crosscore = Counter()
    for event in raw:
        if event['ph'] == 's':
            if event['id'] in pending:
                raise ValueError('Overlapping reused flow ID')
            pending[event['id']] = event
        elif event['ph'] == 'f':
            source = pending.pop(event['id'])
            a, b = event_key(source), event_key(event)
            producer, consumer = work[a], work[b]
            true = source['cat'] == 'true'
            incoming[b].append((a, true))
            row = dict(data_dependency=true,
                       producer_card=producer['card'], producer_core=producer['core'],
                       producer_engine=producer['engine'], producer_kind=producer['kind'],
                       producer_node=producer['node'], producer_oc=producer['oc'],
                       producer_port=producer['port'], producer_end_ms=producer['end_us']/1000,
                       consumer_card=consumer['card'], consumer_core=consumer['core'],
                       consumer_engine=consumer['engine'], consumer_kind=consumer['kind'],
                       consumer_node=consumer['node'], consumer_oc=consumer['oc'],
                       consumer_start_ms=consumer['start_us']/1000)
            flows.append(row)
            if true and producer['card'] == consumer['card'] >= 0 and producer['core'] != consumer['core']:
                crosscore[producer['card'], producer['core'], consumer['core']] += 1
    if pending:
        raise ValueError('Unmatched dependency starts')

    p2p = [r for r in work.values() if r['port'] and ('->' in r['port'] or '<-' in r['port'])]
    transactions = defaultdict(dict)
    for r in p2p:
        match = re.fullmatch(r'(\d+)(->|<-)(\d+)', r['port'])
        if not match:
            raise ValueError(f'Unsupported multicast port {r["port"]}')
        a, direction, b = int(match[1]), match[2], int(match[3])
        src, dst = (a, b) if direction == '->' else (b, a)
        key = src, dst, r['transaction']
        label = 'send' if direction == '->' else 'receive'
        if label in transactions[key]:
            raise ValueError('Repeated P2P endpoint')
        transactions[key][label] = r
    for pair in transactions.values():
        if set(pair) != {'send', 'receive'} or pair['send']['bytes'] != pair['receive']['bytes']:
            raise ValueError('P2P send/receive payload mismatch')
    p2p_summary = []
    for node in sorted({r['node'] for r in p2p}):
        rows = [p['send'] for p in transactions.values() if p['send']['node'] == node]
        p2p_summary.append(dict(node=node, transfers=len(rows), bytes=sum(r['bytes'] for r in rows),
                                start_ms=min(r['start_us'] for r in rows)/1000,
                                end_ms=max(r['end_us'] for r in rows)/1000))

    node_groups = defaultdict(list)
    for r in work.values():
        if r['card'] >= 0:
            node_groups[r['node'], r['engine'], r['kind'], r['memory']].append(r)
    nodes = []
    for (node, engine, kind, memory), rows in sorted(node_groups.items()):
        core_busy = defaultdict(float)
        for r in rows:
            core_busy[r['card'], r['core']] += r['duration_us']/1000
        nodes.append(dict(node=node, engine=engine, kind=kind, memory=memory, events=len(rows),
            cores=len(core_busy), median_active_core_ms=statistics.median(core_busy.values()),
            max_core_ms=max(core_busy.values()), sum_core_ms=sum(core_busy.values()),
            start_ms=min(r['start_us'] for r in rows)/1000, end_ms=max(r['end_us'] for r in rows)/1000,
            descriptor_output_bytes=sum(r['bytes'] for r in rows)))

    projection_copies = [r for r in work.values() if r['kind'] == 'aiccopytovtcm' and
                   r['memory'] == 'DDR' and r['node'].removeprefix(STEM) in COLD]
    # Repeated, uniform projection tiles total exactly the logical trained weights.
    # C128 also has a nonuniform 384 KiB copy on card 3/core 15; preserve it as
    # unclassified rather than quietly charging every MatMul-attributed DMA to weights.
    tile_sizes = {n: (262144 if capacity == 128 and n != 'MatMul_5' else 1572864) for n in COLD}
    cold_copies = [r for r in projection_copies if r['bytes'] == tile_sizes[r['node'].removeprefix(STEM)]]
    other_copies = [r for r in projection_copies if r not in cold_copies]
    if sum(r['bytes'] for r in cold_copies) != 64*3*2048*768*2:
        raise ValueError('Cold projection copy payload differs from logical weight size')
    if len({(r['card'], r['core']) for r in cold_copies}) != 64:
        raise ValueError('Missing cold projection DDR copies')
    copy_counts = Counter((r['card'], r['core'], r['node'].removeprefix(STEM)) for r in cold_copies)
    for card in range(4):
        for core in range(16):
            for node in COLD:
                expected = 12 if capacity == 128 and node != 'MatMul_5' else 2
                if copy_counts[card, core, node] != expected:
                    raise ValueError('Nonuniform projection-copy tiling')

    # Record every direct producer for each core's first cold down GEMM, plus all cold
    # GEMMs on a fixed representative core. Include scheduling edges, labeled separately.
    examples = []
    for card in range(4):
        for core in range(16):
            candidates = [r for r in hmx if r['card'] == card and r['core'] == core and r['node'] == STEM+'MatMul_5']
            target = min(candidates, key=lambda r: r['start_us'])
            examples.append(target)
    examples.extend(r for r in hmx if r['card'] == 0 and r['core'] == 0 and r['node'].removeprefix(STEM) in COLD)
    producers = []
    seen = set()
    for target in examples:
        key = target['pid'], target['tid'], target['oc']
        if key in seen:
            continue
        seen.add(key)
        for source, true in incoming[key]:
            producer = work[source]
            producers.append(dict(target_card=target['card'], target_core=target['core'],
                target_node=target['node'], target_oc=target['oc'], target_start_ms=target['start_us']/1000,
                data_dependency=true, producer_card=producer['card'], producer_core=producer['core'],
                producer_engine=producer['engine'], producer_kind=producer['kind'],
                producer_node=producer['node'], producer_oc=producer['oc'], producer_port=producer['port'],
                producer_start_ms=producer['start_us']/1000, producer_end_ms=producer['end_us']/1000,
                producer_bytes=producer['bytes'], visible_gap_us=target['start_us']-producer['end_us']))

    prefix = out/f'sample{sample}'
    prefix.mkdir(exist_ok=True)
    for name, rows in [('work', list(work.values())), ('waits', waits), ('cores', core_rows),
                       ('nodes', nodes), ('p2p', p2p), ('p2p_summary', p2p_summary),
                       ('flows', flows), ('producers', producers), ('cold_copies', cold_copies)]:
        csv_write(prefix/f'{name}.csv', rows)
    csv_write(prefix/'cross_core.csv', [dict(card=c, producer_core=a, consumer_core=b, true_dependencies=n)
                                      for (c, a, b), n in sorted(crosscore.items())])
    cold_cores = [r for r in core_rows if r['stage'] == 'cold']
    result = dict(capacity=capacity, sample=sample, trace=str(path), trace_sha256=sha(path),
                  device_ms=(finish-origin)/1000, device_start_ms=origin/1000, phases=phases,
                  phase_boundaries_ms=[x/1000 for x in edges],
                  cold_compute_median_core_ms=statistics.median(r['compute_ms'] for r in cold_cores),
                  cold_wait_median_core_ms=statistics.median(r['wait_ms'] for r in cold_cores),
                  cold_first_down_spread_ms=max(r['first_down_ms'] for r in cold_cores)-min(r['first_down_ms'] for r in cold_cores),
                  cold_copy_bytes=sum(r['bytes'] for r in cold_copies), cold_copy_events=len(cold_copies),
                  unclassified_projection_copy_bytes=sum(r['bytes'] for r in other_copies),
                  unclassified_projection_copies=other_copies,
                  flow_pairs=len(flows), true_cross_core_dependencies=sum(crosscore.values()),
                  p2p_transactions=len(transactions), p2p_send_bytes=sum(p['send']['bytes'] for p in transactions.values()),
                  max_wait_pair_error_us=max(r['wait_pair_error_us'] for r in core_rows))
    write(prefix/'summary.json', result)
    print(f'C{capacity} sample {sample}: {result["device_ms"]:.3f} ms, {len(flows)} matched flow pairs', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--sdk', type=Path, default=Path('/opt/qti-aic'))
    p.add_argument('--decode', action='store_true', help='Re-decode saved raw captures with full dependency flows')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.sdk/'exec/qaic-opstats'
    raw_schema, cls = schema_from_sdk(binary)
    (args.out/'AICOpstatsDesc.proto.pb').write_bytes(raw_schema)
    cases = []
    for capacity in [128, 2]:
        out = args.out/f'c{capacity}'
        out.mkdir(exist_ok=True)
        source = args.cold/f'c{capacity}'
        validation = read(source/'profile/validation.json')
        if not validation['bit_exact'] or validation['samples'] != 3:
            raise ValueError('Need three validated captures per case')
        shutil.copy2(source/'profile/validation.json', out/'output_validation.json')
        if args.decode:
            cmd = [str(binary), '--qpc', str(args.scratch/f'c{capacity}_compile/qpc/programqpc.bin'),
                   '--input-dir', str(source/'profile/stats'), '--output-dir', str(out/'trace'),
                   '--trace', '--summary', '--merge-mq-traces', 'true', '--flow-events', 'full']
            write(out/'decode_command.json', cmd)
            with (out/'decode.log').open('w') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        lookup = metadata(args, capacity, cls)
        paths = sorted((out/'trace').glob('*merged*.trace.json'))
        if len(paths) != 3:
            raise ValueError('Expected three full-flow merged traces; use --decode')
        samples = [analyze(path, capacity, index, lookup, out) for index, path in enumerate(paths)]
        chosen = sorted(samples, key=lambda r: r['device_ms'])[1]
        reference = read(source/'analysis/summary.json')
        if abs(chosen['device_ms']-reference['device_ms']) > 1e-8 or chosen['sample'] != reference['sample']:
            raise ValueError('Dependency decode differs from validated original trace')
        chosen = dict(chosen, samples=samples, host_median_ms=reference['host_median_ms'])
        cases.append(chosen)
    with (args.profile/'breakdown/layers.csv').open() as stream:
        full = [r for r in csv.DictReader(stream) if r['layer'] == '2' and r['precision'] == 'fp16']
    if len(full) != 3:
        raise ValueError('Need native/min/pow2 full-model layer-2 context')
    csv_write(args.out/'full_model_layer2.csv', full)
    write(args.out/'summary.json', dict(layer=2, cases=cases, full_model=full,
        full_model_source=str(args.profile/'breakdown/layers.csv'),
        full_model_source_sha256=sha(args.profile/'breakdown/layers.csv'),
        decoder=str(binary), decoder_sha256=sha(binary),
        schema_sha256=hashlib.sha256(raw_schema).hexdigest(),
        validation='Six saved bit-exact captures; no device execution in this analysis'))


if __name__ == '__main__':
    main()
