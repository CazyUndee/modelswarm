from modelswarm.client import Client

c = Client()

# Get full experiment details - check raw fields
exps = c.get_experiments(competition_id='playground-series-s6e8')
print('Raw experiment keys:', exps[0].keys() if exps else 'none')
print()
for e in exps:
    print(e)
    print()
