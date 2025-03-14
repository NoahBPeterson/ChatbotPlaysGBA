# Pokémon Game Agent: AI-Powered Gameplay

The `GameAgent` is an AI-powered system that plays Pokémon games through a continuous loop of:
1. **Observing** the game state (via screenshots and memory reading)
2. **Thinking** about the optimal moves (using LLM providers like OpenAI, Anthropic, or Google Gemini)
3. **Acting** by executing button presses and game commands
4. **Learning** by maintaining context of previous interactions

## How It Works

### 1. Game State Observation
The agent captures the current game state through two primary methods:
- **Screenshots**: Using the mGBA-http API to capture the current frame
- **Memory Reading**: Accessing the game's memory to retrieve additional state data

This combined approach provides a rich understanding of the game state, incorporating both visual and internal game data.

### 2. LLM-Powered Decision Making
The captured game state is sent to a Language Learning Model (LLM) with a structured prompt that includes:
- Current location and player position
- Objects and NPCs in the vicinity
- Available exits and paths
- Memory data like map ID

The LLM analyzes this information and decides which actions to take next.

### 3. Command Execution
The agent parses the LLM's response to extract commands, which can include:
- Button presses (A, B, START, UP, etc.)
- Button sequences for complex actions
- Navigation to specific locations
- Puzzle solving instructions

### 4. Learning Loop
The agent maintains a conversation history with the LLM, allowing it to reference past actions and their outcomes when making new decisions.

## Technical Architecture

The `GameAgent` is built with a modular architecture:

### Core Components
- **MGBAController**: Interfaces with the mGBA emulator via HTTP API
- **VisionController**: Provides computer vision analysis of the game screen
- **LLM Providers**: Abstract interfaces to various LLM services

### LLM Provider Support
The agent supports multiple LLM providers through a unified interface:
- **OpenAI**: GPT-4o and other models
- **Anthropic**: Claude models
- **Google Gemini**: Pro and Flash models

### Command System
The agent includes a flexible command system allowing the LLM to:
- `press_button`: Press a single button momentarily
- `press_sequence`: Execute a series of button presses
- `hold_button` / `release_button`: For continuous input
- `read_memory`: Access specific memory locations
- `navigate_to`: High-level navigation to locations
- `solve_puzzle`: Specialized puzzle-solving logic

### Logging and Analytics
All interactions are recorded to a session log file that includes:
- Timestamped screenshots of each game state
- LLM prompts and responses
- Executed commands and their results
- Game progress metrics

## Setup and Usage

### Prerequisites
- Python 3.6+
- Running mGBA-http server
- API keys for at least one LLM provider:
  - OpenAI API key
  - Anthropic API key
  - Google Gemini API key

### API Key Configuration
You can set up API keys in two ways:
1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GEMINI_API_KEY=your_gemini_key_here
   ```

2. Pass the API key directly when initializing the agent:
   ```python
   agent = GameAgent(llm_provider="openai", api_key="your_key_here")
   ```

### Running the Agent

Basic usage:
```python
from game_agent import GameAgent

# Initialize with default settings (OpenAI provider)
agent = GameAgent()

# Run for 10 steps
agent.run_loop(steps=10)

# Or run indefinitely
# agent.run_loop()
```

Advanced configuration:
```python
# Use Anthropic's Claude
agent = GameAgent(
    llm_provider="anthropic", 
    model="claude-3-opus-20240229",
    history_length=15,
    screenshot_dir="my_screenshots"
)

# Run with custom parameters
agent.run_loop(steps=0, max_errors=5, delay=2.0)
```

### Command-Line Usage
The agent can also be run directly from the command line:

```bash
# Run with Gemini provider for 10 steps
./game_agent.py --provider gemini --steps 10

# Use a specific model
./game_agent.py --provider openai --model gpt-4-turbo --steps 5
```

## Example Session Log

Here's an example interaction from a session log:

```json
{
  "timestamp": "2024-03-09T14:32:45.123456",
  "game_state": {
    "location": "Title Screen",
    "map_id": 0,
    "screenshot_path": "agent_screenshots/screenshot_20240309_143245.png"
  },
  "prompt": "# Game State Analysis - POKEMON EMER (AGB-BPEE)\n...",
  "llm_response": "I can see we're at the Title Screen. Let's start the game.\n\npress_button:START",
  "commands": [{"command": "press_button", "params": "START"}],
  "results": [{"command": "press_button", "params": "START", "result": "Button START pressed successfully", "success": true}]
}
```

## Customization and Extension

The GameAgent is designed to be extended in several ways:

1. **Adding New Commands**: Implement new command handler methods and add them to `self.available_commands`

2. **Custom LLM Providers**: Create new provider classes by extending `LLMProvider` and implementing the `generate_response` method

3. **Enhanced Game State**: Add new data sources to the `_capture_game_state` method to provide richer context

4. **Domain-Specific Prompt Engineering**: Customize the prompt format in `_format_prompt` to improve LLM performance

## Debugging

For debugging issues, check the following:

1. The `logs` directory for detailed session logs
2. The `agent_screenshots` directory (or your custom screenshot directory) for captured frames
3. The console output for real-time logging information

## Limitations

- The agent's performance depends heavily on the quality of the LLM responses
- There may be a delay between observation and action due to API call latency
- Memory reading capabilities are game-specific and may require customization
- The solution is optimized for Pokémon games and may need adaptation for other games

## Advanced Features

### Adaptive Model Selection

The GameAgent now features an intelligent adaptive model selection system that balances cost and performance:

1. **Cost-Effective Starting Point**: By default, the agent starts with cheaper, faster models:
   - OpenAI: gpt-4o-mini
   - Anthropic: claude-3-haiku
   - Gemini: gemini-1.5-flash
   - DeepSeek: deepseek-chat

2. **Smart Escalation**: If the agent detects it's stuck (no movement for 5 consecutive steps), it automatically switches to more powerful reasoning models:
   - OpenAI: gpt-4o
   - Anthropic: claude-3-7-sonnet
   - Gemini: gemini-1.5-pro
   - DeepSeek: (remains on deepseek-chat as alternatives aren't available)

3. **Progress Tracking**: The agent continuously monitors the player's position to determine if meaningful progress is being made in the game.

This feature ensures the agent uses cost-effective models for straightforward scenarios, while automatically engaging more powerful models for complex situations that require deeper reasoning.

To disable adaptive model selection, use the `--no-adaptive-models` flag:
```
./game_agent.py --provider openai --no-adaptive-models
```

### Function Calling Integration

The GameAgent now supports function calling with all major LLM providers:

- **OpenAI**: Using the latest tools API with structured outputs and strict schema validation
- **Anthropic**: Leveraging Claude 3.7's tool use capabilities with extended thinking
- **Gemini**: Utilizing Google's function calling interface

This function calling integration enables more structured and reliable interactions, allowing the LLMs to make precise, well-typed requests for game actions. Instead of generating free-form text that needs to be parsed, the models now generate structured function calls that directly map to game commands.

Benefits of function calling integration:
- More reliable action generation
- Stricter parameter validation
- Better error handling
- More consistent responses across different LLM providers

### Claude 3.7 Extended Thinking

The GameAgent supports Claude 3.7's extended thinking capabilities, which allows the model to perform much more thorough analysis of complex game situations before making decisions. This is particularly useful for:

- Analyzing complex puzzles with multiple potential solutions
- Planning multi-step sequences in battles or navigation challenges
- Considering the long-term implications of game choices

When using the `anthropic` provider with Claude 3.7, the system automatically enables extended thinking with a token budget of 8,000 tokens. This means Claude can perform detailed reasoning before deciding on actions.

Example of using Claude 3.7 with extended thinking:

```bash
# Run with Claude 3.7 Sonnet for complex decision making
./game_agent.py --provider anthropic --model claude-3-7-sonnet-20250219
```

The extended thinking is visible in the logs, where you'll see Claude's step-by-step reasoning process before it recommends actions.

### Hybrid Memory and Vision Approach

The GameAgent uses a hybrid approach combining both memory reading and visual analysis to make decisions. This allows the agent to:

- **Memory Reading**: Access game memory to retrieve additional state data
- **Visual Analysis**: Use computer vision to analyze the game screen

This combination provides a rich understanding of the game state, incorporating both internal game data and visual cues.

## Provider Consistency

The GameAgent now ensures consistency between text and vision models when using the same provider. This means:

1. **OpenAI**: Uses GPT-4o for text interactions and GPT-4o Vision for image analysis.
2. **Anthropic**: Uses Claude for text interactions and Claude's vision capabilities for image analysis.
3. **Gemini**: Uses Gemini models for both text and vision.
4. **DeepSeek**: Uses DeepSeek models for text, and falls back to Gemini for vision (as DeepSeek currently lacks vision capabilities).

This consistency ensures:
- More coherent understanding between what the agent sees and how it responds
- Less confusion when transitioning between visual and text analysis
- Consistent quality of analysis and decision-making

To use a specific provider, use the `--provider` flag:
```
./test_game_agent.py --provider openai --steps 10
./test_game_agent.py --provider anthropic --steps 10
./test_game_agent.py --provider gemini --steps 10
./test_game_agent.py --provider deepseek --steps 10
```

### Special Handling for Title Screens

The GameAgent includes special handling for title screens to ensure the agent can progress past them consistently. When the vision system identifies the current screen as a title screen, it provides explicit instructions to press START or A to begin the game. This automatic handling works across all supported providers. 