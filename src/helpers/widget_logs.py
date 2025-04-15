import os
import time
import threading
import ipywidgets as widgets
from IPython.display import display
from src.helpers.logs import wait_for_container, get_container_logs


def start_log_viewer(
    service_filter: str, 
    wait_time: int = 30, 
    refresh_interval: float = 2.0, 
    tail: int = 50
) -> None:
    """
    Launches a live-updating log viewer widget for a container matching service_filter.
    
    Initially only the "Start Refresh" button is shown. When clicked, logs are streamed
    in a text area and the "Start Refresh" button is hidden, with the "Stop Refresh" button shown.
    When "Stop Refresh" is clicked, log streaming stops and the buttons toggle back.
    
    Args:
        service_filter (str): Substring used to match container names (e.g. "sagemaker" or "robomaker").
        wait_time (int): Maximum seconds to wait for the container.
        refresh_interval (float): Seconds between log refreshes.
        tail (int): Number of last log lines to fetch on each refresh.
    """
    container_id = wait_for_container(service_filter, wait_time=wait_time)
    
    if not container_id:
        error_label = widgets.Label(
            value=f"No container matching '{service_filter}' was found after {wait_time} seconds."
        )
        display(error_label)
        return

    log_area = widgets.Textarea(
        value="",
        description=f"Logs ({service_filter}):",
        layout=widgets.Layout(width="100%", height="500px")
    )

    start_button = widgets.Button(
        description="Start Refresh", 
        button_style="success"
    )
    stop_button = widgets.Button(
        description="Stop Refresh", 
        button_style="danger"
    )
    stop_button.layout.display = "none"

    buttons_box = widgets.HBox([start_button, stop_button])
    display(log_area, buttons_box)

    stop_flag = {"value": False}
    log_thread = {"thread": None}

    def refresh_logs():
        while not stop_flag["value"]:
            log_area.value = get_container_logs(container_id, tail=tail)
            time.sleep(refresh_interval)

    def on_start_click(_):
        start_button.layout.display = "none"
        stop_button.layout.display = ""
        stop_flag["value"] = False
        log_area.value = "Starting log refresh...\n"
        if not log_thread.get("thread") or not log_thread["thread"].is_alive():
            thread = threading.Thread(target=refresh_logs, daemon=True)
            log_thread["thread"] = thread
            thread.start()

    def on_stop_click(_):
        stop_flag["value"] = True
        log_area.value += "\n\nLog refresh stopped."
        start_button.layout.display = ""
        stop_button.layout.display = "none"

    start_button.on_click(on_start_click)
    stop_button.on_click(on_stop_click)