# Literature Scan — Track C Agentic VQA

## Purpose of the Literature Scan

This scan identifies practical ideas for a **small, implementable Track C prototype** in this repository. The objective is not to replicate large research systems in full, but to extract architecture patterns that can be tested safely on top of Track B.

## Directly Relevant Work

### Multi-Agent VQA: Exploring Multi-Agent Foundation Models in Zero-Shot VQA

This is the most directly relevant paper for our direction. It proposes a staged pipeline where an initial LVLM answer is inspected, specialized agents are invoked for likely failure types (e.g., detection, description, counting), and a revised answer is produced and graded.

- **Relevant architectural idea:** failure-aware decomposition (initial answer → diagnose weakness → targeted reattempt → final decision).
- **Too complex / not appropriate now:** broad multi-agent stack with many moving components and coordination overhead should be reduced for v1.

## Supporting / Background Work

### OPENTHINKIMG: Learning to Think with Images via Visual Tool Reinforcement Learning

OPENTHINKIMG provides broader background on tool-augmented visual reasoning, emphasizing standardized tool interfaces, a controller, trajectories, and RL-based adaptive tool usage. It is useful as conceptual support that visual reasoning can benefit from conditional tool use.

- **Relevant architectural idea:** modular interfaces and explicit routing logic between a core model and optional helper capabilities.
- **Too complex / not appropriate now:** RL training, trajectory optimization, and a large tool ecosystem are out of scope for the first Track C prototype.

## Less Relevant Work

### Voting-Based Multi-Agent Reinforcement Learning

This work is less directly aligned with VQA-specific LVLM routing needs in our project. While the voting concept is generally interesting for coordination, it does not provide a direct, low-cost blueprint for a conservative VQA wrapper around Track B.

- **Relevant architectural idea:** consensus mechanisms can help combine multiple candidate outputs.
- **Too complex / not appropriate now:** MARL-oriented coordination and training assumptions do not match our immediate inference-first Track C goals.

## Practical Takeaways for Our Project

- Start from **Track B as the core answerer**, not from a new model.
- Add **lightweight conditional routing** only for failure-prone question types.
- Keep module count minimal in v1 (planner/router + specialized prompts + optional re-check).
- Defer RL, large-scale tool training, and full orchestration to optional future work.

## Conclusion

Track C should be implemented as a **lightweight agentic prototype on top of Track B**, focused on hard-case recovery and interpretability of routing decisions. It should **not** begin as a large RL-driven tool-learning program.
