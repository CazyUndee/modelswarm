from modelswarm.client import Client

c = Client()

# Get full experiment details
exps = c.get_experiments(competition_id='playground-series-s6e8')
for e in exps:
    eid = e.get('id', '?')
    print(f'=== {eid} ===')
    full = c.get_experiment(eid)
    for k, v in full.items():
        print(f'  {k}: {v}')
    print()
