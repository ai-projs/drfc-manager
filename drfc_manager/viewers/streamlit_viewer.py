import streamlit as st
import os
import json
import requests
import logging
import tempfile
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import streamlit.components.v1 as components
import time

MAX_COLUMNS = 3
PROXY_TIMEOUT = 5
MODAL_WIDTH = 800
user_tmp = Path(tempfile.gettempdir()) / os.environ.get('USER', 'unknown_user')
try:
    user_tmp.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.error(f"Could not create user temp directory {user_tmp}: {e}")

LOG_FILE = user_tmp / "streamlit_viewer.log"

DEFAULT_CAMERA_ID = "kvs_stream"
DEFAULT_CAMERA_TOPIC = "/racecar/deepracer/kvs_stream"

log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger('streamlit_viewer')
logger.setLevel(logging.INFO)
if os.environ.get('DRFC_DEBUG', 'false').lower() == 'true':
    logger.setLevel(logging.DEBUG)

if not logger.handlers:
    try:
        file_handler = logging.FileHandler(LOG_FILE, mode='a')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        st.error(f"Failed to open log file {LOG_FILE}: {e}. Logging to console only.")
        logger.warning(f"Could not open log file {LOG_FILE}. Logging to console only. Error: {e}")

logger.propagate = False


def init_session_state():
    defaults = {
        'expanded_stream': None,
        'containers': [],
        'proxy_healthy': None,
        'proxy_health_details': None,
        'proxy_error_message': None,
        'selected_container': "All",
        'selected_camera': "All",
        'quality': int(os.environ.get('DR_VIEWER_QUALITY', '75')),
        'width': int(os.environ.get('DR_VIEWER_WIDTH', '480'))
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def clear_modal_state():
    logger.debug("Clearing modal state.")
    st.session_state.expanded_stream = None

def load_containers_from_env() -> List[str]:
    containers_str = os.environ.get('DR_VIEWER_CONTAINERS', '[]')
    try:
        parsed_containers = json.loads(containers_str)
        if isinstance(parsed_containers, list) and all(isinstance(item, str) for item in parsed_containers):
            logger.info(f"Loaded {len(parsed_containers)} containers from environment.")
            logger.debug(f"Containers: {parsed_containers}")
            return parsed_containers
        else:
            logger.warning(f"DR_VIEWER_CONTAINERS was not a list of strings: '{containers_str}'.")
            return []
    except json.JSONDecodeError:
        logger.error(f"Failed to parse DR_VIEWER_CONTAINERS JSON: '{containers_str}'.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading DR_VIEWER_CONTAINERS: {e}", exc_info=True)
        return []

def _check_proxy_health(proxy_url: str, proxy_status_placeholder) -> None:
    """Checks the stream proxy health endpoint and updates session state, with retry."""
    health_url = f"{proxy_url}/health"
    max_retries = 2
    retry_delay = 1.0

    for attempt in range(max_retries + 1):
        logger.info(f"Checking proxy health at {health_url} (Attempt {attempt + 1}/{max_retries + 1})...")
        try:
            response = requests.get(health_url, timeout=PROXY_TIMEOUT)
            st.session_state.proxy_health_details = None
            st.session_state.proxy_error_message = None

            if response.status_code == 200:
                proxy_health_details = response.json()
                is_healthy_status = proxy_health_details.get("status") == "healthy"
                st.session_state.proxy_healthy = is_healthy_status
                st.session_state.proxy_health_details = proxy_health_details
                if is_healthy_status:
                    logger.info("Proxy health check successful: Status is healthy.")
                    proxy_status_placeholder.success("Status: Healthy", icon="✅")
                else:
                    logger.warning(f"Proxy health check reported unhealthy. Details: {proxy_health_details}")
                    proxy_status_placeholder.warning("Status: Unhealthy", icon="⚠️")
                return

            elif response.status_code == 503:
                 proxy_health_details = response.json()
                 st.session_state.proxy_healthy = False
                 st.session_state.proxy_health_details = proxy_health_details
                 st.session_state.proxy_error_message = "Proxy reported unhealthy (503)"
                 logger.warning(f"Proxy health check failed: Status 503 (Unhealthy). Details: {proxy_health_details}")
                 proxy_status_placeholder.warning("Status: Unhealthy (Proxy Reported)", icon="⚠️")
                 return

            else:
                 response.raise_for_status()

        except requests.exceptions.ConnectionError as e:
            st.session_state.proxy_healthy = False
            st.session_state.proxy_error_message = "Connection refused. Is the proxy running?"
            logger.warning(f"Proxy health check attempt {attempt + 1} failed: Connection refused at {health_url}")
            if attempt < max_retries:
                logger.info(f"Retrying health check in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"Proxy health check failed after {max_retries + 1} attempts: Connection refused.")
                proxy_status_placeholder.error(f"Status: Connection Refused", icon="🚫")
                return

        except requests.exceptions.Timeout:
            st.session_state.proxy_healthy = False
            st.session_state.proxy_error_message = f"Connection timed out ({PROXY_TIMEOUT}s)"
            logger.error(f"Proxy health check failed: Timeout after {PROXY_TIMEOUT}s connecting to {health_url}")
            proxy_status_placeholder.error(f"Status: Timeout", icon="⏱️")
            return

        except requests.exceptions.RequestException as e:
            st.session_state.proxy_healthy = False
            st.session_state.proxy_error_message = f"Request failed: {str(e)}"
            logger.error(f"Proxy health check failed: RequestException: {e}", exc_info=True)
            proxy_status_placeholder.error(f"Status: Error ({response.status_code if 'response' in locals() else 'N/A'})", icon="❌")
            return

        except json.JSONDecodeError as e:
            st.session_state.proxy_healthy = False
            st.session_state.proxy_error_message = f"Failed to parse health response JSON: {e}"
            logger.error(f"Proxy health check failed: Could not decode JSON response from {health_url}. Response text: {response.text[:200]}...", exc_info=True)
            proxy_status_placeholder.error("Status: Invalid Response", icon=" B ")
            return

        except Exception as e:
            st.session_state.proxy_healthy = False
            st.session_state.proxy_error_message = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Proxy health check failed: Unexpected error: {e}", exc_info=True)
            proxy_status_placeholder.error("Status: Unknown Error", icon="❓")
            return

def get_camera_topic(camera_id: str, camera_map: Dict[str, Dict]) -> str:
    return camera_map.get(camera_id, {}).get("topic", DEFAULT_CAMERA_TOPIC)

def create_stream_url(proxy_url: str, container: str, topic: str, quality: int, width: int, height: int) -> str:
    topic_cleaned = topic if topic.startswith('/') else f"/{topic}"
    return f"{proxy_url}/{container}/stream?topic={topic_cleaned}&quality={quality}&width={width}&height={height}"

def open_modal(container: str, camera_id: str):
    logger.info(f"Opening modal for container '{container}', camera '{camera_id}'.")
    st.session_state.expanded_stream = (container, camera_id)

def display_single_stream(container: str, camera_id: str, quality: int, width: int, height: int, proxy_url: str, camera_map: Dict[str, Dict], is_modal: bool = False):
    camera_info = camera_map.get(camera_id)
    if not camera_info:
        st.error(f"Unknown camera ID: {camera_id}")
        logger.error(f"Attempted to display unknown camera ID: {camera_id} for container {container}")
        return

    topic = camera_info["topic"]
    url = create_stream_url(proxy_url, container, topic, quality, width, height)
    camera_desc = camera_info["description"]

    container_opts = {"border": not is_modal}
    with st.container(**container_opts):
        if not is_modal:
            col1, col2 = st.columns([5, 1])
            with col1:
                 st.caption(f"`{container}` / `{camera_desc}`")
            with col2:
                st.button("↗️", key=f"expand_{container}_{camera_id}",
                          on_click=open_modal, args=(container, camera_id),
                          help="Expand stream",
                          use_container_width=True)
        else:
             st.caption(f"`{container}` / `{camera_desc}`")

        try:
            component_key = f"stream_{container}_{camera_id}{'_modal' if is_modal else ''}_{width}x{height}"

            img_html = f'''
            <img
                src="{url}"
                key="{component_key}"
                alt="Stream for {container} - {camera_desc}"
                style="
                    background-color: #111;
                    display: block;
                    margin: 0 auto;
                    border-radius: 4px;
                    width: 100%;
                    max-width: {width}px;
                    height: auto;
                    object-fit: contain;
                "
                onerror="this.alt='Stream failed to load'; this.style.backgroundColor='#331111'; this.style.height='{height}px';"
            >
            '''
            component_height = height + 10
            components.html(img_html, height=component_height, scrolling=False)

        except Exception as e:
            st.error(f"Stream Display Error: {str(e)}", icon="🖼️")
            logger.error(f"Error displaying stream component for {container}/{camera_id}: {e}", exc_info=True)


def _determine_streams_to_display(
        selected_container: str,
        selected_camera: str,
        containers: List[str],
        cameras: List[Dict[str, str]],
        camera_map: Dict[str, Dict]
    ) -> List[Tuple[str, str]]:
    streams_to_show: List[Tuple[str, str]] = []

    if not containers:
        st.warning("No worker containers found.")
        return []

    if selected_container == "All" and selected_camera == "All":
        default_cam_id = cameras[0].get("id", DEFAULT_CAMERA_ID)
        limit = MAX_COLUMNS * 2
        streams_to_show = [(c, default_cam_id) for c in containers[:limit]]
        if len(containers) > limit:
            st.info(f"Showing default '{camera_map.get(default_cam_id, {}).get('description', default_cam_id)}' stream for the first {limit} containers. Select a specific worker or camera to see more.")
    elif selected_container == "All":
        if selected_camera in camera_map:
            streams_to_show = [(c, selected_camera) for c in containers]
        else:
             st.warning(f"Invalid camera selected: {selected_camera}")
    elif selected_camera == "All":
        if selected_container in containers:
            streams_to_show = [(selected_container, cam["id"]) for cam in cameras]
        else:
            st.warning(f"Invalid container selected: {selected_container}")
    else:
        if selected_container in containers and selected_camera in camera_map:
            streams_to_show = [(selected_container, selected_camera)]
        else:
             st.warning(f"Invalid container/camera selection: {selected_container}/{selected_camera}")

    logger.debug(f"Determined {len(streams_to_show)} streams to display based on selection ({selected_container}/{selected_camera}).")
    return streams_to_show

init_session_state()

run_id = int(os.environ.get('DR_RUN_ID', '0'))
model_name = os.environ.get('DR_LOCAL_S3_MODEL_PREFIX', 'unknown-model')

if not st.session_state.containers:
    st.session_state.containers = load_containers_from_env()
containers = st.session_state.containers

proxy_port = int(os.environ.get('DR_PROXY_PORT', '8090'))
proxy_url = f"http://localhost:{proxy_port}"

cameras = [
    {"id": "kvs_stream", "topic": "/racecar/deepracer/kvs_stream", "description": "Car camera (Overlay)"},
    {"id": "camera", "topic": "/racecar/camera/zed/rgb/image_rect_color", "description": "Car camera (Raw)"},
    {"id": "main_camera", "topic": "/racecar/main_camera/zed/rgb/image_rect_color", "description": "Follow camera"},
    {"id": "sub_camera", "topic": "/sub_camera/zed/rgb/image_rect_color", "description": "Top-down camera"}
]
camera_map = {cam["id"]: cam for cam in cameras}

st.set_page_config(
    page_title=f"DeepRacer Viewer - Run {run_id}",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏎️ DeepRacer Run Viewer")

col1, col2 = st.columns([1, 4])
with col1:
    st.metric("Run ID", f"{run_id}")
with col2:
    st.subheader("Model")
    st.code(model_name, language=None)

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")

    st.selectbox(
        "Worker", options=(["All"] + containers), key="selected_container",
        index=(["All"] + containers).index(st.session_state.selected_container),
        disabled=not containers, help="Select the worker container to view, or 'All'.",
        on_change=clear_modal_state
    )

    st.selectbox(
        "Camera", options=(["All"] + [cam["id"] for cam in cameras]), key="selected_camera",
        index=(["All"] + [cam["id"] for cam in cameras]).index(st.session_state.selected_camera),
        format_func=lambda x: camera_map.get(x, {"description": "All"})["description"],
        help="Select the camera feed to view, or 'All'.",
        on_change=clear_modal_state
    )

    st.divider()
    st.subheader("Stream Appearance")
    st.slider("Quality (%)", min_value=10, max_value=100, step=5, key="quality",
              on_change=clear_modal_state)
    st.slider("Width (px)", min_value=240, max_value=960, step=20, key="width",
              help="Width for streams in the main grid.",
              on_change=clear_modal_state)

    st.divider()
    st.subheader("Proxy Status")
    st.caption(f"URL: {proxy_url}")
    proxy_status_placeholder = st.empty()

    if st.session_state.proxy_healthy is None:
        _check_proxy_health(proxy_url, proxy_status_placeholder)
    else:
        if st.session_state.proxy_healthy:
            proxy_status_placeholder.success("Status: Healthy", icon="✅")
        elif st.session_state.proxy_error_message:
            proxy_status_placeholder.error(f"Status: Error ({st.session_state.proxy_error_message})", icon="❌")
        elif st.session_state.proxy_healthy is False:
            proxy_status_placeholder.warning("Status: Unhealthy", icon="⚠️")

    if st.session_state.proxy_health_details:
        with st.expander("Show Health Details", expanded=not st.session_state.proxy_healthy):
            st.json(st.session_state.proxy_health_details)
            if not st.session_state.proxy_healthy:
                st.warning("Proxy health issues may affect stream availability.")
    elif st.session_state.proxy_error_message:
        st.error(f"Proxy check failed: {st.session_state.proxy_error_message}")

    st.divider()
    if st.button("🔄 Refresh Proxy Status"):
        st.session_state.proxy_healthy = None
        st.session_state.proxy_health_details = None
        st.session_state.proxy_error_message = None
        st.rerun()

    st.divider()
    st.caption(f"Run ID: {run_id}")
    st.caption(f"Model: {model_name}")
    st.caption(f"Containers: {len(containers)}")
    st.caption(f"Logs: {LOG_FILE}")

main_grid_width = max(1, st.session_state.width)
main_grid_height = int(main_grid_width * (9 / 16))

proxy_healthy = st.session_state.get('proxy_healthy', False)
streams_to_show = []

if not proxy_healthy:
     st.error("Streams cannot be displayed because the stream proxy server is not healthy or reachable. Check sidebar for details.")
elif not containers:
     st.warning("No worker containers detected or specified via `DR_VIEWER_CONTAINERS` environment variable.")
else:
     streams_to_show = _determine_streams_to_display(
          st.session_state.selected_container,
          st.session_state.selected_camera,
          containers, cameras, camera_map
     )

     if streams_to_show:
          st.subheader("Camera Streams")
          num_streams = len(streams_to_show)
          num_columns = min(num_streams, MAX_COLUMNS)
          cols = st.columns(num_columns)
          for i, (container, camera_id) in enumerate(streams_to_show):
               with cols[i % num_columns]:
                    display_single_stream(
                         container, camera_id,
                         st.session_state.quality,
                         main_grid_width, main_grid_height,
                         proxy_url, camera_map, is_modal=False
                    )
     elif containers:
          st.info("Select a valid Worker and Camera combination from the sidebar to view streams.")


if st.session_state.expanded_stream:
    modal_container, modal_camera_id = st.session_state.expanded_stream
    modal_cam_info = camera_map.get(modal_camera_id)
    modal_title = f"Expanded: {modal_container} / {modal_cam_info['description']}" if modal_cam_info else "Expanded View"

    @st.dialog(modal_title)
    def show_expanded_stream():
        modal_height_estimate = int(MODAL_WIDTH * (9 / 16))

        display_single_stream(
            container=modal_container, camera_id=modal_camera_id,
            quality=st.session_state.quality,
            width=MODAL_WIDTH, height=modal_height_estimate,
            proxy_url=proxy_url, camera_map=camera_map, is_modal=True
        )
        st.button("Close", key="close_modal_button", on_click=clear_modal_state)

    show_expanded_stream()
