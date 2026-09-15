# intinerary-RL
This is a simple [OpenEnv](https://github.com/huggingface/openenv) and Docker environment for evaluating RL strategies. There are no trained policies to be found here, only ca random agent with support for a heuristic-based greedy algorithm. 

I made this environment via conversation with a mix of Cursor Grok 4.6, Claude Opus 5, GPT 5.6-Sol, Codex 5.3, and Claude Opus 4.5. Please note that Sol has very rarely been known to permanently erase data, so its use is not broadly advised.

The specifications and contract for v1 can be found in [docs/spec.md](docs/spec.md). The base environment for a flight itinerary recommendation system has been implemented and minimally validated with just one test case, which is a flight from JFK -> LAX. Most active time was spent on design, with every (incremental) build test passing as codebase components were added. No design decisions were revisited during the build phase. 

Instructions for usage will be added shortly. 

Please see [planned extensions for v2](docs/spec.md).
