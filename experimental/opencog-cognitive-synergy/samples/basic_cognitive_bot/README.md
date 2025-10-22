# Basic Cognitive Bot Sample

This sample demonstrates a simple bot enhanced with OpenCog cognitive architecture capabilities.

## Features

- **OpenCog AtomSpace**: Knowledge representation and reasoning
- **Autonomous Decision Making**: Bot makes independent decisions using cognitive reasoning
- **Continuous Learning**: Adapts behavior based on interactions
- **Self-Modification**: Can modify its own structure through autogenesis
- **Cognitive Status Monitoring**: Real-time insight into cognitive processes

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the bot:
   ```bash
   python app.py
   ```

3. Test with Bot Framework Emulator or connect to channels

## Endpoints

- **Messages**: `POST /api/messages` - Main bot endpoint
- **Health Check**: `GET /health` - Basic health and cognitive status
- **Cognitive Status**: `GET /cognitive-status` - Detailed cognitive state information

## Cognitive Architecture

The bot uses several cognitive components:

### AtomSpace
- Knowledge stored as atoms and links
- Concepts, predicates, and relationships
- Truth values for probabilistic reasoning

### Cognitive Engine
- Reasoning in multiple modes (deductive, inductive, abductive, analogical, creative)
- Goal-directed behavior
- Attention management
- Learning from experience

### Autogenesis System
- Self-assessment and improvement
- Autonomous architecture modification
- Performance optimization
- Emergent capability development

## Example Interactions

Try asking about:
- "What are your cognitive capabilities?"
- "How do you learn and adapt?"
- "Tell me about autonomous thinking"
- "What is cognitive synergy?"
- "How does your reasoning work?"

The bot will engage with cognitive reasoning and provide insights into its own thinking processes.

## Monitoring

Check the cognitive status endpoint to see:
- Active goals and attention focus
- Energy and confidence levels
- Atomspace size and complexity
- Recent self-modifications
- Performance trends

## Configuration

Set environment variables:
- `MicrosoftAppId` - Bot Framework App ID
- `MicrosoftAppPassword` - Bot Framework App Password
- `MicrosoftAppType` - App type (default: MultiTenant)
- `MicrosoftAppTenantId` - Tenant ID (for SingleTenant apps)

## Experimental Status

⚠️ This is experimental code designed for research and demonstration of cognitive architectures in conversational AI.