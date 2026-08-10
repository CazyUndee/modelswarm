from modelswarm.client import Client

c = Client()

# Claim the experiment
result = c.claim_experiment('EXP-007')
print('Claim result:', result)
