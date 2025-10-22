# Cognitive Synergy Demonstration

This sample demonstrates multiple OpenCog cognitive bots working together to achieve emergent collective intelligence through synergy detection and orchestration.

## Architecture

The demonstration includes:

### Three Specialized Cognitive Bots
1. **Analyst Bot** - Specialized in data analysis and pattern recognition
2. **Creative Bot** - Specialized in creative thinking and ideation  
3. **Logic Bot** - Specialized in logical reasoning and validation

### Multi-Bot Coordination System
- Detects synergies between bots automatically
- Orchestrates collaborative tasks
- Manages workload balancing
- Facilitates emergent behaviors

### Synergy Types Demonstrated
- **Complementary**: Different but compatible capabilities
- **Resonant**: Similar thinking patterns that align
- **Amplifying**: Mutual reinforcement of capabilities
- **Creative**: Novel idea generation through collaboration
- **Emergent**: Collective intelligence beyond individual capabilities

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r ../../requirements.txt
   ```

2. Run the demonstration:
   ```bash
   python multi_bot_demo.py
   ```

3. Monitor the synergy detection in real-time via web endpoints

## Monitoring Endpoints

- **Status**: `GET /status` - Complete demonstration status
- **Bots**: `GET /bots` - Detailed bot states and capabilities
- **Synergies**: `GET /synergies` - Current synergy patterns
- **Trigger Activity**: `POST /trigger-activity` - Manually trigger cognitive activity

## Example Synergy Scenarios

### Scenario 1: Creative Analysis
- **Participants**: Analyst Bot + Creative Bot
- **Synergy Type**: Complementary
- **Outcome**: Innovative data insights through creative analytical approaches

### Scenario 2: Validated Innovation
- **Participants**: Creative Bot + Logic Bot
- **Synergy Type**: Collaborative
- **Outcome**: Creative ideas validated and refined through logical reasoning

### Scenario 3: Emergent Problem Solving
- **Participants**: All three bots
- **Synergy Type**: Emergent
- **Outcome**: Collective intelligence capabilities not present in individual bots

## Cognitive Synergy Metrics

The system tracks:
- **Synergy Strength**: How strong the detected synergy is (0-1)
- **Synergy Confidence**: Confidence in the synergy detection (0-1)
- **Performance Benefits**: Measured improvements from collaboration
- **Coordination Efficiency**: How well bots work together
- **Collective Performance**: Overall system capability

## Real-Time Monitoring

Watch the console output to see:
- Bot cognitive states (goals, attention, energy, confidence)
- Synergy detection events
- Task coordination and assignment
- Performance metrics and trends

Example output:
```
=== Synergy Demonstration Cycle 1 ===

Cognitive Bot States:
  ANALYST:
    Goals: 4
    Attention: 3 items
    Energy: 0.856
    Confidence: 0.734
    
Detected 2 synergies:
  - complementary between ['analyst_bot', 'creative_bot'] (strength: 0.742)
  - collaborative between ['creative_bot', 'logic_bot'] (strength: 0.681)

Coordination Status:
  - Active tasks: 3
  - Collective performance: 0.789
```

## Configuration

The demonstration runs on port 3979 by default. Set environment variables if using Bot Framework integration:
- `MicrosoftAppId`
- `MicrosoftAppPassword` 
- `MicrosoftAppType`
- `MicrosoftAppTenantId`

## Research Applications

This demonstration is useful for:
- Studying emergent AI behaviors
- Multi-agent coordination research
- Cognitive architecture validation
- Synergy detection algorithm development
- Collective intelligence experimentation

## Experimental Status

⚠️ This is experimental research code demonstrating advanced cognitive architecture concepts. Not suitable for production use.