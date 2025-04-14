from typing import Callable, TypeVar, Any, Dict, Optional
import inspect
import time
from threading import Lock

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(
    parent_container: DeltaGenerator,
    update_interval: float = 0.1,  # Time in seconds between UI updates
    buffer_size: int = 10,  # Number of tokens to buffer before updating
    auto_scroll: bool = True,  # Whether to auto-scroll to the bottom
    scroll_frequency: int = 15,  # Only scroll every N updates
) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
        update_interval (float): Minimum time in seconds between UI updates (default: 0.1).
        buffer_size (int): Number of tokens to buffer before forcing an update (default: 10).
        auto_scroll (bool): Whether to auto-scroll to the bottom on updates (default: True).
        scroll_frequency (int): Only scroll every N updates to reduce scrolling frequency (default: 5).
        
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a custom callback handler class for managing and displaying stream events in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
        Includes improvements for smoother UI updates and controlled scrolling behavior.
        """

        def __init__(
            self, 
            container: st.delta_generator.DeltaGenerator, 
            initial_text: str = "",
            update_interval: float = update_interval,
            buffer_size: int = buffer_size,
            auto_scroll: bool = auto_scroll,
            scroll_frequency: int = scroll_frequency,
        ):
            """
            Initializes the StreamHandler with a Streamlit container and optional initial text.
            
            Args:
                container (st.delta_generator.DeltaGenerator): The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to start with in the container.
                update_interval (float): Minimum time in seconds between UI updates.
                buffer_size (int): Number of tokens to collect before forcing an update.
                auto_scroll (bool): Whether to auto-scroll to the bottom on updates.
                scroll_frequency (int): Only scroll every N updates to reduce scrolling frequency.
            """
            self.container = container  # The Streamlit container to update
            self.thoughts_placeholder = self.container.container()  # container to hold tool_call renders
            self.tool_output_placeholder = None  # placeholder for the output of the tool call to be in the expander
            self.token_placeholder = self.container.empty()  # for token streaming
            self.text = initial_text  # The text content to display, starting with initial text
            
            # Throttling and buffering parameters
            self.update_interval = update_interval
            self.buffer_size = buffer_size
            self.auto_scroll = auto_scroll
            self.scroll_frequency = max(1, scroll_frequency)  # Ensure at least 1
            
            # State variables for throttling and buffering
            self.token_buffer = []
            self.last_update_time = time.time()
            self.buffer_lock = Lock()  # Thread safety for buffer operations
            self.update_count = 0  # Track number of updates for debugging and scroll control

        def _should_update(self) -> bool:
            """
            Determines if the UI should be updated based on buffer size and time elapsed.
            
            Returns:
                bool: True if an update should occur, False otherwise.
            """
            current_time = time.time()
            time_elapsed = current_time - self.last_update_time
            
            # Update if either condition is met:
            # 1. Buffer size threshold reached
            # 2. Minimum time interval has passed and we have tokens to display
            return (len(self.token_buffer) >= self.buffer_size or 
                   (time_elapsed >= self.update_interval and len(self.token_buffer) > 0))

        def _update_ui(self) -> None:
            """
            Updates the UI with the current text content.
            Uses a more efficient approach to reduce jitter.
            """
            # Apply buffered tokens to the text
            with self.buffer_lock:
                if self.token_buffer:
                    self.text += ''.join(self.token_buffer)
                    self.token_buffer = []
                    self.last_update_time = time.time()
                    self.update_count += 1
            
            # Update the UI with the current text
            self.token_placeholder.markdown(self.text)
            
            # Auto-scrolling feature has been removed
            # The UI will now focus on user prompts when provided

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """
            Callback method triggered when a new token is received (e.g., from a language model).
            Buffers tokens and updates the UI based on throttling parameters.
            
            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            # Add the token to the buffer
            with self.buffer_lock:
                self.token_buffer.append(token)
            
            # Check if we should update the UI
            if self._should_update():
                self._update_ui()

        def on_llm_end(self, response, **kwargs) -> None:
            """
            Callback method triggered when the LLM response is complete.
            Ensures any remaining buffered tokens are displayed.
            
            Args:
                response: The final response object.
                **kwargs: Additional keyword arguments.
            """
            # Force an update to display any remaining buffered tokens
            self._update_ui()
            
            # Auto-scrolling feature has been removed
            # The UI will now focus on user prompts when provided

        def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
            """
            Run when the tool starts running.
            
            Args:
                serialized (Dict[str, Any]): The serialized tool.
                input_str (str): The input string.
                kwargs (Any): Additional keyword arguments.
            """
            # Force an update to display any buffered tokens before showing tool output
            self._update_ui()
            
            with self.thoughts_placeholder:
                status_placeholder = st.empty()   # Placeholder to show the tool's status
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("called ", serialized["name"])  # Show which tool is being called
                    st.write("tool description: ", serialized["description"])
                    st.write("tool input: ")
                    st.code(input_str)   # Display the input data sent to the tool
                    st.write("tool output: ")
                    # Placeholder for tool output that will be updated later below
                    self.tool_output_placeholder = st.empty()
                    s.update(label="Completed Calling Tool!", expanded=False)   # Update the status once done

        def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
            """
            Run when the tool ends.
            
            Args:
                output (Any): The output from the tool.
                kwargs (Any): Additional keyword arguments.
            """
            # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
            if self.tool_output_placeholder:
                self.tool_output_placeholder.code(output.content)   # Display the tool's output

    # Define a type variable for generic type hinting in the decorator, to maintain
    # input function and wrapped function return type
    fn_return_type = TypeVar('fn_return_type')

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        
        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.
            
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        ctx = get_script_run_ctx()  # Retrieve the current Streamlit script execution context

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            
            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
                
            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(ctx=ctx)  # Add the Streamlit context to the current execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(
        parent_container,
        update_interval=update_interval,
        buffer_size=buffer_size,
        auto_scroll=auto_scroll,
        scroll_frequency=scroll_frequency,
    )

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):  # Identify callback methods
            setattr(st_cb, method_name, add_streamlit_context(method_func))  # Wrap and replace the method

    # Return the fully configured StreamHandler instance with the context-aware callback methods
    return st_cb
