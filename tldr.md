# tldr 
## What It Does

This is a **cost predictor for a multi-model LLM system**. It predicts how much money you'll spend running queries through a routing system that sends different query types to different AI models.

## The System You're Modeling

```
User Query → Classifier → Routes to best model
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
           GEMINI          CODER             GROK
         (visual)         (code)          (research)
              ↓               ↓               ↓
         Can delegate to other models if needed
```

**The flow:**
1. User sends a query
2. A cheap **classifier** model decides: is this visual, code, or research?
3. Query gets routed to the appropriate **lead model**
4. If the query is complex, the lead model may **delegate** to another model
5. Each step costs tokens → tokens cost money

## What The Predictor Does

**Input:** Usage profile defining:
- Query mix (20% visual, 30% code, 50% research)
- Complexity distribution (60% simple, 30% medium, 10% complex)
- Routing accuracy (85% correct routing)
- Daily query volume (1000 queries/day)

**Process:** Monte Carlo simulation
- Simulates thousands of queries
- Randomly assigns types/complexity based on distributions
- Simulates routing mistakes (15% go to wrong model → costs more)
- Simulates delegations for complex queries
- Calculates token costs at each step

**Output:**
- Monthly cost estimate
- P95/P99 cost (worst-case planning)
- Cost breakdown (routing vs lead vs delegation)
- Sensitivity analysis (what if routing accuracy drops?)

## Why It's Useful

Instead of guessing costs, you can:
- Compare scenarios: "What if we get 50% more code queries?"
- Plan budgets: "P99 monthly cost is $X, so budget for that"
- Optimize: "Improving routing accuracy from 70% to 90% saves $Y/month"
- Use real data: Feed your Jarvis logs to validate/calibrate the model