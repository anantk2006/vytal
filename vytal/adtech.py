import numpy as np
from scipy import stats
import csv
from typing import Dict, List, Tuple
import logging

def analyze_eye_tracking_data(results, aois, fps, fixation_threshold_sec=0.5, distance_threshold=50):
    """
    Analyze eye tracking data to calculate metrics for Areas of Interest (AOIs) and general viewing behavior.

    This function processes a series of eye gaze predictions and calculates various metrics
    for predefined Areas of Interest (AOIs) as well as general viewing metrics.

    Parameters:
    results (list of dict): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                            representing the predicted x and y coordinates of the eye gaze.
    aois (dict): A dictionary where keys are AOI names and values are tuples representing
                 the bounding rectangle of each AOI in the format (x1, y1, x2, y2).
    fps (int): The frames per second of the recorded eye tracking data.
    fixation_threshold_sec (float): Minimum duration in seconds for a gaze point to be considered a fixation. Default is 0.5 seconds.
    distance_threshold (float): Maximum distance in pixels between consecutive gaze points to be considered part of the same fixation. Default is 50 pixels.

    Returns:
    tuple: A tuple containing two dictionaries:
           1. aoi_metrics: A dictionary with metrics for each AOI:
              - 'TFF' (Time to First Fixation): Time in seconds before the AOI was first looked at.
              - 'Fixation_Count': Number of fixations on the AOI.
              - 'Total_Fixation_Duration': Total time in seconds spent looking at the AOI.
              - 'Avg_Fixation_Duration': Average duration of fixations on the AOI in seconds.
              - 'Revisits': Number of times the gaze returned to the AOI after looking elsewhere.
           2. general_metrics: A dictionary with general viewing metrics:
              - 'Entry_Point': The coordinates (x, y) where the gaze first entered the stimulus.
              - 'Exit_Point': The coordinates (x, y) where the gaze last left the stimulus.

    Note:
    - This function assumes that the eye tracking data points are equally spaced in time.
    - The fixation detection uses a simple distance-based threshold method.
    """
    if not isinstance(results, list) or not results:
        raise ValueError("Results must be a non-empty list of dictionaries.")
    
    if not isinstance(aois, dict) or not aois:
        raise ValueError("AOIs must be a non-empty dictionary.")
    
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("FPS must be a positive integer.")
    
    if not isinstance(fixation_threshold_sec, (int, float)) or fixation_threshold_sec <= 0:
        raise ValueError("Fixation threshold must be a positive number.")
    
    if not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0:
        raise ValueError("Distance threshold must be a positive number.")
    
    for i, result in enumerate(results):
        if not isinstance(result, dict) or 'pred_x' not in result or 'pred_y' not in result:
            raise ValueError(f"Result at index {i} is not a valid dictionary with 'pred_x' and 'pred_y' keys.")
        if not isinstance(result['pred_x'], (int, float)) or not isinstance(result['pred_y'], (int, float)):
            raise ValueError(f"Result at index {i} contains non-numeric 'pred_x' or 'pred_y' values.")
    
    for aoi_name, aoi_rect in aois.items():
        if not isinstance(aoi_rect, tuple) or len(aoi_rect) != 4:
            raise ValueError(f"AOI '{aoi_name}' does not have a valid rectangle tuple (x1, y1, x2, y2).")
        if not all(isinstance(coord, (int, float)) for coord in aoi_rect):
            raise ValueError(f"AOI '{aoi_name}' contains non-numeric coordinates.")
    
    fixation_threshold_frames = int(fixation_threshold_sec * fps)

    def point_in_rect(x, y, rect):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    aoi_metrics = {aoi_name: {
        'TFF': None,
        'Fixation_Duration': [],
        'Fixation_Count': 0,
        'Total_Fixation_Duration': 0,
        'Revisits': 0
    } for aoi_name in aois}

    general_metrics = {
        'Entry_Point': (results[0]['pred_x'], results[0]['pred_y']),
        'Exit_Point': (results[-1]['pred_x'], results[-1]['pred_y'])
    }

    current_fixation = None
    last_aoi = None

    for i, result in enumerate(results):
        x, y = result['pred_x'], result['pred_y']
        timestamp_sec = i / fps

        for aoi_name, aoi_rect in aois.items():
            if point_in_rect(x, y, aoi_rect):
                if aoi_metrics[aoi_name]['TFF'] is None:
                    aoi_metrics[aoi_name]['TFF'] = timestamp_sec
                if current_fixation is None:
                    current_fixation = (x, y, i)
                elif i == len(results)-1 or i - current_fixation[2] >= fixation_threshold_frames or \
                        (i < len(results) - 1 and
                         ((results[i + 1]['pred_x'] - x) ** 2 + (results[i + 1]['pred_y'] - y) ** 2) ** 0.5 > distance_threshold):
                    fixation_duration_sec = (i - current_fixation[2]) / fps
                    aoi_metrics[aoi_name]['Fixation_Duration'].append(fixation_duration_sec)
                    aoi_metrics[aoi_name]['Fixation_Count'] += 1
                    aoi_metrics[aoi_name]['Total_Fixation_Duration'] += fixation_duration_sec

                    if last_aoi != aoi_name:
                        aoi_metrics[aoi_name]['Revisits'] += 1

                    current_fixation = None

                last_aoi = aoi_name
                break
        else:
            if current_fixation is not None:
                current_fixation = None
            last_aoi = None

    for aoi_name in aoi_metrics:
        if aoi_metrics[aoi_name]['Fixation_Count'] > 0:
            aoi_metrics[aoi_name]['Avg_Fixation_Duration'] = (
                aoi_metrics[aoi_name]['Total_Fixation_Duration'] / aoi_metrics[aoi_name]['Fixation_Count']
            )
        else:
            aoi_metrics[aoi_name]['Avg_Fixation_Duration'] = 0

        del aoi_metrics[aoi_name]['Fixation_Duration']

    return aoi_metrics, general_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, Button, TextBox
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DraggableRectangle:
    """
    A class to make a Matplotlib rectangle draggable along with its associated text.

    Attributes:
    rect (matplotlib.patches.Rectangle): The rectangle object to be made draggable.
    name_text (matplotlib.text.Text): The text object associated with the rectangle.
    press (tuple or None): Stores the initial position of the rectangle and the mouse click coordinates.
    """

    def __init__(self, rect, name_text):
        """
        Initializes the DraggableRectangle class with the given rectangle and associated text.

        Parameters:
        rect (matplotlib.patches.Rectangle): The rectangle object to be made draggable.
        name_text (matplotlib.text.Text): The text object associated with the rectangle.
        """
        if not isinstance(rect, patches.Rectangle):
            raise ValueError("rect must be an instance of matplotlib.patches.Rectangle")
        if not isinstance(name_text, plt.Text):
            raise ValueError("name_text must be an instance of matplotlib.text.Text")
        
        self.rect = rect
        self.name_text = name_text
        self.press = None
        self.text_offset = (name_text.get_position()[0] - rect.get_x(),
                            name_text.get_position()[1] - rect.get_y())
        self.connect()

    def connect(self):
        """Connects the event handlers for mouse interactions."""
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """
        Event handler for mouse button press.

        Parameters:
        event (matplotlib.backend_bases.MouseEvent): The mouse event.
        """
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        self.press = self.rect.xy, (event.xdata, event.ydata)

    def on_motion(self, event):
        """
        Event handler for mouse motion (dragging).

        Parameters:
        event (matplotlib.backend_bases.MouseEvent): The mouse event.
        """
        if self.press is None:
            return
        if event.inaxes != self.rect.axes:
            return
        xy, (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        new_x = xy[0] + dx
        new_y = xy[1] + dy
        self.rect.set_xy((new_x, new_y))
        self.name_text.set_position((new_x + self.text_offset[0], new_y + self.text_offset[1]))
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        """
        Event handler for mouse button release.

        Parameters:
        event (matplotlib.backend_bases.MouseEvent): The mouse event.
        """
        self.press = None
        self.rect.figure.canvas.draw()

def define_aois(image_path: str) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Provides an interactive interface for defining Areas of Interest (AOIs) on an image.

    This function opens a matplotlib window displaying the specified image and allows
    the user to create, select, rename, move, and delete AOIs using mouse interactions
    and GUI buttons.

    Args:
    image_path (str): Path to the image file on which AOIs will be defined.

    Returns:
    Dict[str, Tuple[float, float, float, float]]: A dictionary where keys are AOI names
    and values are tuples representing the bounding box of each AOI in the format
    (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the
    bottom-right corner of the AOI.

    Functionality:
    - Create Mode: Left-click and drag to create a new AOI.
    - Select Mode: Click on an existing AOI to select it.
    - Rename: Type a new name in the text box and click 'Rename' to rename the selected AOI.
    - Delete: Click 'Delete' to remove the selected AOI.
    - Move: Click and drag an existing AOI to move it.
    - Mode Toggle: Use the 'Mode' button to switch between 'Create' and 'Select' modes.
    - Display AOIs: Press 'd' key to display current AOIs in the console.
    - Quit: Press 'q' key or click 'Close' button to finish and close the window.

    Note:
    - The function will return an empty dictionary if there's an error reading the image file.
    - AOIs are represented as rectangles on the image.
    - The function uses matplotlib for rendering and interaction.

    Raises:
    FileNotFoundError: If the specified image file is not found.
    Exception: For any other error occurring while reading the image file.
    """

    try:
        img = plt.imread(image_path)
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return {}

    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure height
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
    ax.imshow(img)

    aois = {}
    draggable_rects = {}
    selected_aoi = [None]
    mode = ['create']

    def print_instructions():
        print("\nInstructions:")
        print("- Use the 'Mode' button to switch between 'Create' and 'Select' modes")
        print("- In 'Create' mode, left-click and drag to create a new AOI")
        print("- In 'Select' mode, click on an AOI to select it")
        print("- Type a new name in the text box and click 'Rename' to rename the selected AOI")
        print("- Click 'Delete' to remove the selected AOI")
        print("- Click and drag an AOI to move it (works in both modes)")
        print("- Press 'd' to display current AOIs")
        print("- Press 'q' to finish and quit\n")

    def onselect(eclick, erelease):
        if mode[0] == 'create':
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if None in [x1, y1, x2, y2]:  # Check if coordinates are valid
                logger.error("Invalid coordinates detected during AOI creation.")
                return
            temp_name = f"AOI_{len(aois) + 1}"
            aois[temp_name] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            rect = patches.Rectangle((min(x1, x2), min(y1, y2)), abs(x2 - x1), abs(y2 - y1),
                                     fill=False, edgecolor='r')
            ax.add_patch(rect)
            text = ax.text(min(x1, x2), min(y1, y2), temp_name, color='r')
            draggable_rects[temp_name] = DraggableRectangle(rect, text)
            fig.canvas.draw()
            print(f"Created {temp_name}. Select it and use 'Rename' to change its name.")

    def onclick(event):
        if event.inaxes != ax or mode[0] != 'select':
            return
        for aoi_name, (x1, y1, x2, y2) in list(aois.items()):
            if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2:
                selected_aoi[0] = aoi_name
                print(f"Selected {aoi_name}")
                rename_textbox.set_val(aoi_name)
                break
        else:
            selected_aoi[0] = None
            rename_textbox.set_val('')

    def rename_aoi(event):
        if selected_aoi[0]:
            new_name = rename_textbox.text
            if new_name and new_name != selected_aoi[0]:
                if new_name in aois:
                    print(f"AOI name {new_name} already exists. Choose a different name.")
                    return
                aois[new_name] = aois.pop(selected_aoi[0])
                draggable_rects[new_name] = draggable_rects.pop(selected_aoi[0])
                draggable_rects[new_name].name_text.set_text(new_name)
                fig.canvas.draw()
                print(f"Renamed {selected_aoi[0]} to {new_name}")
                selected_aoi[0] = new_name

    def delete_aoi(event):
        if selected_aoi[0]:
            del aois[selected_aoi[0]]
            draggable_rects[selected_aoi[0]].rect.remove()
            draggable_rects[selected_aoi[0]].name_text.remove()
            del draggable_rects[selected_aoi[0]]
            fig.canvas.draw()
            print(f"Deleted {selected_aoi[0]}")
            selected_aoi[0] = None
            rename_textbox.set_val('')

    def toggle_mode(event):
        mode[0] = 'select' if mode[0] == 'create' else 'create'
        mode_button.label.set_text(f"Mode: {mode[0].capitalize()}")
        fig.canvas.draw()
        print(f"Switched to {mode[0]} mode")

    def onkey(event):
        if event.key == 'd':
            print("Current AOIs:")
            for name, coords in aois.items():
                print(f"{name}: {coords}")
        elif event.key == 'q':
            plt.close(fig)

    print_instructions()
    rs = RectangleSelector(ax, onselect, useblit=True,
                           button=[1], minspanx=5, minspany=5,
                           spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    # Adjust button and textbox positions
    fig.subplots_adjust(bottom=0.2)  # Increase bottom margin for controls

    button_width = 0.12
    button_height = 0.05
    button_bottom = 0.05
    spacing = 0.02
    textbox_label_width = 0.08  # Width for the TextBox label

    ax_mode = plt.axes([0.02, button_bottom, button_width, button_height])
    ax_textbox = plt.axes([0.02 + button_width + spacing + textbox_label_width, button_bottom, 0.25, button_height])
    ax_rename = plt.axes(
        [0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing, button_bottom, button_width,
         button_height])
    ax_delete = plt.axes(
        [0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing + button_width + spacing, button_bottom,
         button_width, button_height])
    ax_close = plt.axes([
                            0.02 + button_width + spacing + textbox_label_width + 0.25 + spacing + button_width + spacing + button_width + spacing,
                            button_bottom, button_width, button_height])

    mode_button = Button(ax_mode, 'Mode: Create')
    rename_textbox = TextBox(ax_textbox, 'New Name: ')
    btn_rename = Button(ax_rename, 'Rename')
    btn_delete = Button(ax_delete, 'Delete')
    btn_close = Button(ax_close, 'Close')

    # Add close button functionality
    def close_figure(event):
        plt.close(fig)

    btn_close.on_clicked(close_figure)

    mode_button.on_clicked(toggle_mode)
    btn_rename.on_clicked(rename_aoi)
    btn_delete.on_clicked(delete_aoi)

    plt.show()

    # Update aois with final positions of draggable rectangles
    for name, drect in draggable_rects.items():
        x, y = drect.rect.get_xy()
        w, h = drect.rect.get_width(), drect.rect.get_height()
        aois[name] = (x, y, x + w, y + h)

    return aois


def plot_gaze_path(results: List[Dict[str, float]], aois: Dict[str, Tuple[float, float, float, float]],
                   image_path: str):
    """
    Visualizes the gaze path over the advertisement image.

    This function creates a plot showing the path of the viewer's gaze overlaid on the original image,
    along with the defined Areas of Interest (AOIs).

    Args:
    results (List[Dict[str, float]]): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                                      representing the predicted x and y coordinates of the eye gaze.
    aois (Dict[str, Tuple[float, float, float, float]]): A dictionary where keys are AOI names and values
                                                         are tuples representing the bounding box of each AOI
                                                         in the format (x1, y1, x2, y2).
    image_path (str): Path to the image file used as the background for the visualization.

    The function will:
    1. Load and display the background image.
    2. Plot the gaze path as a continuous line.
    3. Overlay scatter points representing individual gaze positions.
    4. Draw rectangles representing the AOIs.

    Note:
    - The gaze path is plotted in blue with low opacity for clarity.
    - The scatter points are colored according to their temporal order using a 'cool' colormap.
    - AOIs are drawn as red rectangles with their names labeled.

    Raises:
    FileNotFoundError: If the specified image file is not found.
    Exception: For any other error occurring while reading the image file.
    """

    try:
        img = plt.imread(image_path)
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return

    # Create a plot with the background image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Extract x and y coordinates from the results
    x = [r['pred_x'] for r in results]
    y = [r['pred_y'] for r in results]

    # Plot the gaze path as a blue line with low opacity
    ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)

    # Overlay scatter points representing individual gaze positions
    ax.scatter(x, y, c=range(len(x)), cmap='cool', s=10, zorder=2)

    # Draw rectangles representing the AOIs
    for aoi_name, (x1, y1, x2, y2) in aois.items():
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r')
        ax.add_patch(rect)
        ax.text(x1, y1, aoi_name, color='r')

    # Set the title and display the plot
    plt.title("Gaze Path Visualization")
    plt.show()


def generate_heatmap(results: List[Dict[str, float]], image_path: str, bins: int = 50):
    """
    Creates a heatmap of gaze intensity overlaid on the advertisement image.

    This function generates a heatmap visualization of the gaze data, showing areas of high and low
    gaze concentration overlaid on the original image.

    Args:
    results (List[Dict[str, float]]): A list of dictionaries, each containing 'pred_x' and 'pred_y' keys
                                      representing the predicted x and y coordinates of the eye gaze.
    image_path (str): Path to the image file used as the background for the heatmap.
    bins (int): Number of bins to use for the 2D histogram. Default is 50.

    The function will:
    1. Load and display the background image.
    2. Create a 2D histogram of the gaze data.
    3. Overlay the heatmap on the image using a 'hot' colormap with partial transparency.
    4. Add a colorbar to show the intensity scale.

    Note:
    - The function includes error checking for empty results, negative coordinates, and coordinates
      outside the image dimensions.
    - The heatmap uses a 'hot' colormap where red indicates areas of high gaze concentration.

    Raises:
    FileNotFoundError: If the specified image file is not found.
    Exception: For any other error occurring while reading the image file or processing the data.
    """

    if not results:
        logger.error("No gaze data provided for heatmap generation.")
        return

    if any(r['pred_x'] < 0 or r['pred_y'] < 0 for r in results):
        logger.warning("Negative coordinates found in gaze data. This may cause issues with heatmap generation.")

    try:
        img = plt.imread(image_path)
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return

    if any(r['pred_x'] > img.shape[1] or r['pred_y'] > img.shape[0] for r in results):
        logger.warning("Gaze coordinates found outside image dimensions. This may cause issues with heatmap generation.")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    x = [r['pred_x'] for r in results]
    y = [r['pred_y'] for r in results]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, img.shape[1]], [0, img.shape[0]]])
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]

    # Create a heatmap overlay
    heatmap_overlay = ax.imshow(heatmap.T, extent=extent, origin='upper', cmap='hot', alpha=0.5)

    plt.title("Gaze Intensity Heatmap")
    plt.colorbar(heatmap_overlay, label='Gaze Intensity')
    plt.show()


def aoi_significance_test(group1_results: List[Dict[str, float]], group2_results: List[Dict[str, float]],
                          aois: Dict[str, Tuple[float, float, float, float]], test: str = 't-test') -> Dict:
    """
    Performs statistical tests to compare AOI metrics between two groups.

    This function calculates and compares metrics for each Area of Interest (AOI) between two groups
    of gaze data, using either a t-test or Mann-Whitney U test.

    Args:
    group1_results (List[Dict[str, float]]): Gaze data for the first group. Each dict should contain
                                             'pred_x' and 'pred_y' keys for gaze coordinates.
    group2_results (List[Dict[str, float]]): Gaze data for the second group. Same format as group1_results.
    aois (Dict[str, Tuple[float, float, float, float]]): A dictionary where keys are AOI names and values
                                                         are tuples representing the bounding box of each AOI
                                                         in the format (x1, y1, x2, y2).
    test (str): Statistical test to use. Either 't-test' or 'mann-whitney'. Default is 't-test'.

    Returns:
    Dict: A dictionary containing the results of the statistical tests for each AOI. Each AOI entry includes:
          - 'group1_mean': Mean value for group 1
          - 'group2_mean': Mean value for group 2
          - 'statistic': The test statistic
          - 'p_value': The p-value of the test

    The function will:
    1. Calculate the proportion of gaze points within each AOI for both groups.
    2. Perform the specified statistical test to compare these proportions between the groups.
    3. Return the results including means, test statistic, and p-value for each AOI.

    Note:
    - The function assumes that the AOIs and gaze coordinates use the same coordinate system.
    - The choice of test should be based on the nature of your data and experimental design.

    Raises:
    ValueError: If an invalid test type is specified.
    """

    def get_aoi_metrics(results, aois):
        metrics = {aoi: [] for aoi in aois}
        for r in results:
            for aoi, (x1, y1, x2, y2) in aois.items():
                if x1 <= r['pred_x'] <= x2 and y1 <= r['pred_y'] <= y2:
                    metrics[aoi].append(1)
                else:
                    metrics[aoi].append(0)
        return metrics

    # Error handling
    if not group1_results or not group2_results:
        raise ValueError("Both group1_results and group2_results must be non-empty lists.")
    
    if not aois:
        raise ValueError("AOIs dictionary must be non-empty.")

    if test not in ['t-test', 'mann-whitney']:
        raise ValueError("Invalid test type. Use 't-test' or 'mann-whitney'.")

    group1_metrics = get_aoi_metrics(group1_results, aois)
    group2_metrics = get_aoi_metrics(group2_results, aois)

    results = {}
    for aoi in aois:
        if test == 't-test':
            statistic, p_value = stats.ttest_ind(group1_metrics[aoi], group2_metrics[aoi])
        elif test == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(group1_metrics[aoi], group2_metrics[aoi])

        results[aoi] = {
            'group1_mean': np.mean(group1_metrics[aoi]),
            'group2_mean': np.mean(group2_metrics[aoi]),
            'statistic': statistic,
            'p_value': p_value
        }

    return results


def export_metrics_to_csv(aoi_metrics: Dict[str, Dict[str, float]], general_metrics: Dict[str, float], filename: str):
    """
    Exports calculated metrics to a CSV file for further analysis in other software.

    This function takes the metrics calculated for Areas of Interest (AOIs) and general viewing behavior
    and writes them to a CSV file in a structured format.

    Args:
    aoi_metrics (Dict[str, Dict[str, float]]): A nested dictionary where the outer key is the AOI name,
                                               and the inner dictionary contains various metrics as key-value pairs.
    general_metrics (Dict[str, float]): A dictionary of general metrics that apply to the entire viewing session.
    filename (str): The name of the output CSV file, including path if necessary.

    The function will:
    1. Create a new CSV file with the specified filename.
    2. Write AOI metrics, with each row containing the AOI name, metric name, and value.
    3. Write general metrics, with each row containing the metric name and value.

    The CSV structure will be:
    AOI Metrics
    AOI, Metric, Value
    [AOI metrics data]

    General Metrics
    Metric, Value
    [General metrics data]

    Note:
    - If the file already exists, it will be overwritten.
    - The function uses the csv module to ensure proper CSV formatting.

    Raises:
    IOError: If there's an error writing to the file (e.g., permission denied, disk full).
    """

    if not isinstance(aoi_metrics, dict) or not isinstance(general_metrics, dict):
        raise ValueError("aoi_metrics and general_metrics must be dictionaries.")
    
    if not filename.endswith('.csv'):
        raise ValueError("The filename must end with .csv")

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['AOI Metrics'])
            writer.writerow(['AOI', 'Metric', 'Value'])
            for aoi, metrics in aoi_metrics.items():
                for metric, value in metrics.items():
                    writer.writerow([aoi, metric, value])

            writer.writerow([])
            writer.writerow(['General Metrics'])
            writer.writerow(['Metric', 'Value'])
            for metric, value in general_metrics.items():
                writer.writerow([metric, value])

        logger.info(f"Metrics exported to {filename}")
    except IOError as e:
        logger.error(f"Error writing to CSV file: {e}")
        raise

def main():
    # Example usage of the functions
    image_path = "demo_ad.jpg"

    # Step 1: Define AOIs dynamically on the picture
    print("Please define Areas of Interest (AOIs) on the advertisement image.")
    aois = define_aois(image_path)
    print("Defined AOIs:", aois)

    # Step 2: Record a short video for eye tracking (simulated here)
    print("\nSimulating video recording for eye tracking...")
    # In a real scenario, you would record actual video here
    video_duration = 10  # seconds
    fps = 30
    frame_count = video_duration * fps

    # Simulate eye-tracking data
    np.random.seed(42)  # for reproducibility
    simulated_results = [
        {'pred_x': np.random.uniform(0, 1920), 'pred_y': np.random.uniform(0, 1080)}
        for _ in range(frame_count)
    ]

    # Step 3: Analyze eye-tracking data
    print("\nAnalyzing eye-tracking data...")
    aoi_metrics, general_metrics = analyze_eye_tracking_data(simulated_results, aois, fps)

    print("\nAOI Metrics:")
    for aoi, metrics in aoi_metrics.items():
        print(f"{aoi}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\nGeneral Metrics:")
    for metric, value in general_metrics.items():
        print(f"{metric}: {value}")

    # Step 4: Visualize gaze path
    print("\nGenerating gaze path visualization...")
    plot_gaze_path(simulated_results, aois, image_path)

    # Step 5: Generate heatmap
    print("\nGenerating gaze intensity heatmap...")
    generate_heatmap(simulated_results, image_path)

    # Step 6: Perform significance test (simulating two groups)
    print("\nPerforming significance test between two simulated groups...")
    group1_results = simulated_results[:len(simulated_results) // 2]
    group2_results = simulated_results[len(simulated_results) // 2:]

    significance_results = aoi_significance_test(group1_results, group2_results, aois)

    print("\nSignificance Test Results:")
    for aoi, results in significance_results.items():
        print(f"{aoi}:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")

    # Step 7: Export metrics to CSV
    print("\nExporting metrics to CSV...")
    export_metrics_to_csv(aoi_metrics, general_metrics, "eye_tracking_metrics.csv")
    print("Metrics exported to eye_tracking_metrics.csv")

    print("\nEye-tracking analysis pipeline completed!")

    # # Step 2: Display the ad and record a 10-second video
    # print("\nPreparing to record a 10-second video. Please look at the displayed advertisement.")
    # screen = cv2.imread(image_path)
    # screen_height, screen_width = screen.shape[:2]
    #
    # # Prepare video recording
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('gaze_recording.mp4', fourcc, 30.0, (screen_width, screen_height))
    #
    # cv2.namedWindow("Advertisement", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Advertisement", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #
    # # Record video
    # start_time = time.time()
    # while time.time() - start_time < 10:  # Record for 10 seconds
    #     out.write(screen)
    #     cv2.imshow("Advertisement", screen)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # out.release()
    # cv2.destroyAllWindows()
    #
    # print("Video recording completed.")
    #
    # # Step 3: Use Vytal API to get gaze predictions
    # print("\nProcessing the video to obtain gaze predictions...")
    # predictor = vytal.Client()
    # results = predictor.predict_from_video('gaze_recording.mp4')
    #
    # # Convert Vytal results to our format
    # processed_results = []
    # for result in results:
    #     gaze_x = result.x * screen_width
    #     gaze_y = result.y * screen_height
    #     processed_results.append({
    #         'pred_x': gaze_x,
    #         'pred_y': gaze_y
    #     })
    #
    # # Step 4: Analyze eye-tracking data
    # print("\nAnalyzing eye-tracking data...")
    # fps = 30  # Assuming 30 fps video
    # aoi_metrics, general_metrics = analyze_eye_tracking_data(processed_results, aois, fps)
    #
    # print("\nAOI Metrics:")
    # for aoi, metrics in aoi_metrics.items():
    #     print(f"{aoi}:")
    #     for metric, value in metrics.items():
    #         print(f"  {metric}: {value}")
    #
    # print("\nGeneral Metrics:")
    # for metric, value in general_metrics.items():
    #     print(f"{metric}: {value}")
    #
    # # Step 5: Visualize gaze path
    # print("\nGenerating gaze path visualization...")
    # plot_gaze_path(processed_results, aois, image_path)
    #
    # # Step 6: Generate heatmap
    # print("\nGenerating gaze intensity heatmap...")
    # generate_heatmap(processed_results, image_path)
    #
    # # Step 7: Export metrics to CSV
    # print("\nExporting metrics to CSV...")
    # export_metrics_to_csv(aoi_metrics, general_metrics, "eye_tracking_metrics.csv")
    # print("Metrics exported to eye_tracking_metrics.csv")
    #
    # print("\nEye-tracking analysis pipeline completed!")


if __name__ == "__main__":
    main()
