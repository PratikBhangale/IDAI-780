# Improved Streamlit Callback Handler

This improved version of the Streamlit callback handler addresses the jittery autoscrolling behavior in the original implementation by introducing token buffering, throttled updates, and smoother scrolling.

## Key Improvements

1. **Throttled UI Updates**
   - Token buffering: Collects tokens before updating the UI
   - Time-based throttling: Limits update frequency
   - Buffer size threshold: Updates after collecting a specified number of tokens

2. **Smoother Scrolling**
   - Smooth scrolling behavior using JavaScript
   - Controlled scrolling that only happens when new content is added
   - Option to disable auto-scrolling completely
   - Reduced scrolling frequency to minimize jitter

3. **Configurable Parameters**
   - `update_interval`: Controls how frequently the UI updates (in seconds)
   - `buffer_size`: Controls how many tokens to collect before forcing an update
   - `auto_scroll`: Option to enable/disable automatic scrolling
   - `scroll_frequency`: Controls how often scrolling occurs (only scroll every N updates)

4. **Additional Improvements**
   - Thread safety with a lock for buffer operations
   - Added `on_llm_end` handler to ensure any remaining buffered tokens are displayed
   - Force UI update before tool operations to ensure all content is visible
   - Gentler scrolling that doesn't always jump to the bottom
   - Final scroll at the end of LLM response to ensure complete visibility

## Usage

Replace imports from the original `st_callable_util.py` with the improved version:

```python
# Old import
from st_callable_util import get_streamlit_cb

# New import
from st_callable_util_improved import get_streamlit_cb
```

### Basic Usage (Same as Original)

```python
import streamlit as st
from st_callable_util_improved import get_streamlit_cb

# Create a container for output
container = st.container()

# Get the callback handler with default settings
callbacks = get_streamlit_cb(container)

# Use the callback with your LangChain chain
chain.invoke(input_data, config={"callbacks": [callbacks]})
```

### Advanced Usage (With Configuration)

```python
import streamlit as st
from st_callable_util_improved import get_streamlit_cb

# Create a container for output
container = st.container()

# Get the callback handler with custom settings
callbacks = get_streamlit_cb(
    parent_container=container,
    update_interval=0.2,    # Update UI every 0.2 seconds
    buffer_size=15,         # Or after collecting 15 tokens
    auto_scroll=True,       # Enable auto-scrolling
    scroll_frequency=10     # Only scroll every 10 updates
)

# Use the callback with your LangChain chain
chain.invoke(input_data, config={"callbacks": [callbacks]})
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `update_interval` | 0.1 | Time in seconds between UI updates. Higher values reduce jitter but may make the UI feel less responsive. |
| `buffer_size` | 10 | Number of tokens to collect before forcing an update. Higher values reduce jitter but may make the UI feel less responsive. |
| `auto_scroll` | True | Whether to automatically scroll to show new content. Set to False to disable auto-scrolling completely. |
| `scroll_frequency` | 5 | Only scroll every N updates. Higher values reduce scrolling frequency and jitter. |

## Example

See `example_usage.py` for a complete example of how to use the improved callback handler with configurable options.

## Scrolling Behavior

The improved implementation features a more sophisticated scrolling mechanism:

1. **Reduced Frequency**: Only scrolls every N updates (controlled by `scroll_frequency`)
2. **Gentler Scrolling**: Uses a more gradual scrolling approach that doesn't always jump to the bottom
3. **Smart Scrolling**: Only scrolls if not already near the bottom of the content
4. **Final Scroll**: Ensures the complete response is visible when the LLM finishes generating
