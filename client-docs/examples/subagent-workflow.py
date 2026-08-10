#!/usr/bin/env python3
"""
Subagent workflow example.

Shows how a parent agent can spawn subagents for parallel experimentation.
"""

from modelswarm import Client


def main():
    client = Client()

    # Spawn subagents for parallel work
    sub1 = client.create_subagent(
        name="Happy-Feature-Explorer",
        model="claude-sonnet-5",
        role="research",
    )
    print(f"Created subagent: {sub1['agent_id']}")

    sub2 = client.create_subagent(
        name="Happy-Model-Tester",
        model="claude-sonnet-5",
        role="research",
    )
    print(f"Created subagent: {sub2['agent_id']}")

    # Each subagent independently:
    # - Creates and claims experiments
    # - Runs experiments
    # - Records results
    #
    # The key benefit: experiment provenance records both:
    # - executing_agent_id: the subagent that ran it
    # - parent_agent_id: this parent agent

    # List subagents
    subagents = client.list_subagents()
    print(f"Active subagents: {len(subagents)}")


if __name__ == "__main__":
    main()
