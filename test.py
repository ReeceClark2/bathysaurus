import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

CLIENT_NAME = "EARTHSCOPE"

STATIONS = [
    {
        "network": "OO",
        "station": "AXBA1",
        "location": "*",
        "channel": "LHZ",
        "label": "AXBA1",
    },
    {
        "network": "OO",
        "station": "HYS14",
        "location": "*",
        "channel": "LHZ",
        "label": "HYS14",
    },
]

START_TIME = UTCDateTime("2026-09-01T00:00:00")

NUMBER_OF_LUNAR_DAYS = 7

# Lunar/tidal day:
# 24 hours 50 minutes = 89,400 seconds
LUNAR_DAY_SECONDS = 24 * 60 * 60 + 50 * 60
LUNAR_DAY_HOURS = LUNAR_DAY_SECONDS / 3600.0

END_TIME = (
    START_TIME
    + NUMBER_OF_LUNAR_DAYS * LUNAR_DAY_SECONDS
)

# Broad exploratory period range.
#
# This includes:
# - semidiurnal tidal periods near 12 hours
# - diurnal periods near 24 hours
# - shorter-period internal-wave signals
MIN_PERIOD_HOURS = 0.5
MAX_PERIOD_HOURS = 30.0

# Response-corrected output units:
# "ACC"  = acceleration
# "VEL"  = velocity
# "DISP" = displacement
RESPONSE_OUTPUT = "ACC"


# ---------------------------------------------------------------------
# Download and preprocess one station
# ---------------------------------------------------------------------

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException


def observed_data(
    network,
    station,
    channel,
    location,
    start_time,
    end_time,
    response_output="ACC",
):
    """
    Download, merge, response-correct, and resample one station.

    The location may be a specific code such as "00" or a wildcard "*".
    """

    client = Client(CLIENT_NAME)

    request_description = (
        f"{network}.{station}.{location}.{channel} "
        f"from {start_time} to {end_time}"
    )

    print(f"\nRequesting: {request_description}")

    # -------------------------------------------------------------
    # Download response metadata
    # -------------------------------------------------------------

    try:
        inventory = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start_time,
            endtime=end_time,
            level="response",
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not retrieve response metadata for:\n"
            f"  {request_description}\n\n"
            f"Original error:\n{error}"
        ) from error

    if len(inventory.networks) == 0:
        raise RuntimeError(
            f"No station metadata found for:\n"
            f"  {request_description}\n\n"
            f"Try location='*' and verify the station/channel."
        )

    print("Metadata found:")
    print(inventory)

    # -------------------------------------------------------------
    # Download waveform data
    # -------------------------------------------------------------

    try:
        stream = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start_time,
            endtime=end_time,
            attach_response=False,
        )

    except FDSNNoDataException as error:
        print(
            f"\nNo waveform data returned for:\n"
            f"  {request_description}"
        )

        print(
            "\nLikely causes:"
            "\n  1. The station was not operating during this interval."
            "\n  2. The location code is incorrect."
            "\n  3. The requested channel is unavailable."
            "\n  4. The station has metadata but no waveform archive for this span."
        )

        raise RuntimeError(
            f"EarthScope returned HTTP 204: no waveform data for "
            f"{request_description}"
        ) from error

    if len(stream) == 0:
        raise RuntimeError(
            f"EarthScope returned an empty stream for:\n"
            f"  {request_description}"
        )

    print("\nWaveform data returned:")
    print(stream)

    # -------------------------------------------------------------
    # Merge segmented data
    # -------------------------------------------------------------

    stream.merge(
        method=1,
        fill_value="interpolate",
    )

    # Select by network, station, and channel.
    # Do not require a blank location when using location="*".
    selected = stream.select(
        network=network,
        station=station,
        channel=channel,
    )

    if len(selected) == 0:
        raise RuntimeError(
            f"Waveform data were returned, but could not select:\n"
            f"  {network}.{station}.*.{channel}\n\n"
            f"Returned stream was:\n{stream}"
        )

    if len(selected) > 1:
        print(
            "\nMultiple location codes were returned. "
            "Using the first trace:"
        )

    trace = selected[0]

    print(f"Using trace: {trace.id}")

    # -------------------------------------------------------------
    # Trim and preprocess
    # -------------------------------------------------------------

    trace.trim(
        starttime=start_time,
        endtime=end_time,
        pad=True,
        fill_value=0,
    )

    trace.detrend("demean")
    trace.detrend("linear")

    print(f"Removing instrument response from {trace.id}")

    trace.remove_response(
        inventory=inventory,
        output=response_output,
        water_level=60,
        pre_filt=(1e-7, 1e-6, 0.4, 0.49),
    )

    # -------------------------------------------------------------
    # Resample to 1 Hz
    # -------------------------------------------------------------

    if not np.isclose(trace.stats.sampling_rate, 1.0):
        print(
            f"Resampling {trace.id} from "
            f"{trace.stats.sampling_rate:.4f} Hz to 1 Hz"
        )

        trace.resample(
            sampling_rate=1.0,
            no_filter=False,
            window="hann",
        )

    trace.trim(
        starttime=start_time,
        endtime=end_time,
        pad=True,
        fill_value=0,
    )

    trace.detrend("demean")
    trace.detrend("linear")

    return trace


# ---------------------------------------------------------------------
# Apply broad period-band filter
# ---------------------------------------------------------------------

def filter_period_band(
    trace,
    min_period_hours,
    max_period_hours,
):
    """
    Apply a zero-phase bandpass filter using period limits in hours.
    """

    freqmin = 1.0 / (max_period_hours * 3600.0)
    freqmax = 1.0 / (min_period_hours * 3600.0)

    filtered_trace = trace.copy()

    filtered_trace.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=4,
        zerophase=True,
    )

    return filtered_trace, freqmin, freqmax


# ---------------------------------------------------------------------
# Split one trace into lunar-day segments
# ---------------------------------------------------------------------

def make_lunar_day_matrix(
    trace,
    number_of_lunar_days,
    lunar_day_seconds,
):
    """
    Return equal-length lunar-day segments.

    Returns
    -------
    time_hours : ndarray
        Time within one lunar day in hours.

    day_matrix : ndarray
        Shape:
        (number_of_lunar_days, samples_per_lunar_day)
    """

    sampling_rate = float(trace.stats.sampling_rate)
    sample_interval = float(trace.stats.delta)

    if not np.isclose(sampling_rate, 1.0):
        raise RuntimeError(
            f"{trace.id} is not sampled at 1 Hz after resampling."
        )

    samples_per_lunar_day = int(
        round(lunar_day_seconds * sampling_rate)
    )

    required_samples = (
        number_of_lunar_days * samples_per_lunar_day
    )

    data = np.asarray(trace.data, dtype=float)
    data = np.nan_to_num(data)

    if len(data) < required_samples:
        raise RuntimeError(
            f"Trace {trace.id} does not contain enough samples for "
            f"{number_of_lunar_days} complete lunar days.\n"
            f"Required samples: {required_samples:,}\n"
            f"Available samples: {len(data):,}"
        )

    # Use exactly seven complete lunar days.
    data = data[:required_samples]

    # Normalize each station by its maximum absolute amplitude.
    # This avoids one station dominating merely because of units or gain.
    amplitude_scale = np.max(np.abs(data))

    if amplitude_scale == 0.0:
        raise RuntimeError(
            f"Trace {trace.id} contains only zero-valued samples."
        )

    data = data / amplitude_scale

    # Rows are lunar days.
    # Columns are time positions within each lunar day.
    day_matrix = data.reshape(
        number_of_lunar_days,
        samples_per_lunar_day,
    )

    time_hours = (
        np.arange(samples_per_lunar_day)
        * sample_interval
        / 3600.0
    )

    return time_hours, day_matrix


# ---------------------------------------------------------------------
# Process all stations
# ---------------------------------------------------------------------

station_results = []

for station_info in STATIONS:

    network = station_info["network"]
    station = station_info["station"]
    location = station_info["location"]
    channel = station_info["channel"]
    label = station_info["label"]

    trace = observed_data(
        network=network,
        station=station,
        channel=channel,
        location=location,
        start_time=START_TIME,
        end_time=END_TIME,
        response_output=RESPONSE_OUTPUT,
    )

    filtered_trace, freqmin, freqmax = filter_period_band(
        trace=trace,
        min_period_hours=MIN_PERIOD_HOURS,
        max_period_hours=MAX_PERIOD_HOURS,
    )

    time_hours, day_matrix = make_lunar_day_matrix(
        trace=filtered_trace,
        number_of_lunar_days=NUMBER_OF_LUNAR_DAYS,
        lunar_day_seconds=LUNAR_DAY_SECONDS,
    )

    # Pointwise product:
    #
    # product[j] =
    #     day_1[j] * day_2[j] * ... * day_7[j]
    #
    # Because the data were normalized, this product is dimensionless.
    product_all_days = np.sum(
        day_matrix,
        axis=0,
    )

    station_results.append(
        {
            "label": label,
            "trace": filtered_trace,
            "time_hours": time_hours,
            "day_matrix": day_matrix,
            "product": product_all_days,
            "freqmin": freqmin,
            "freqmax": freqmax,
        }
    )

    print(
        f"\nFinal filtered trace for {label}:"
    )
    print(filtered_trace)

    print(
        f"Sampling rate: "
        f"{filtered_trace.stats.sampling_rate:.4f} Hz"
    )

    print(
        f"Samples per lunar day: "
        f"{day_matrix.shape[1]:,}"
    )

    print(
        f"Lunar-day duration: "
        f"{LUNAR_DAY_SECONDS:,} seconds "
        f"({LUNAR_DAY_HOURS:.6f} hours)"
    )

    print(
        f"Total samples used: "
        f"{day_matrix.size:,}"
    )

    print(
        f"Period band: "
        f"{MIN_PERIOD_HOURS:g}–{MAX_PERIOD_HOURS:g} hours"
    )

    print(
        f"Frequency band: "
        f"{freqmin:.8e}–{freqmax:.8e} Hz"
    )

    print(
        f"Product minimum: "
        f"{np.min(product_all_days):.6e}"
    )

    print(
        f"Product maximum: "
        f"{np.max(product_all_days):.6e}"
    )


# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------

number_of_stations = len(station_results)

fig, axes = plt.subplots(
    number_of_stations,
    2,
    figsize=(16, 5 * number_of_stations),
    sharex="col",
    constrained_layout=True,
)

# Make axes consistently two-dimensional when only one station exists.
axes = np.atleast_2d(axes)

colors = plt.cm.viridis(
    np.linspace(
        0.05,
        0.95,
        NUMBER_OF_LUNAR_DAYS,
    )
)

for station_index, result in enumerate(station_results):

    label = result["label"]
    time_hours = result["time_hours"]
    day_matrix = result["day_matrix"]
    product_all_days = result["product"]

    overlay_axis = axes[station_index, 0]
    product_axis = axes[station_index, 1]

    # -------------------------------------------------------------
    # Seven overlaid lunar days
    # -------------------------------------------------------------

    for day_index in range(NUMBER_OF_LUNAR_DAYS):

        overlay_axis.plot(
            time_hours,
            day_matrix[day_index],
            linewidth=0.8,
            color=colors[day_index],
            label=f"Lunar day {day_index + 1}",
        )

    overlay_axis.set_title(
        f"{label}: seven overlaid lunar days\n"
        f"{MIN_PERIOD_HOURS:g}–{MAX_PERIOD_HOURS:g} hour period band"
    )

    overlay_axis.set_ylabel(
        "Normalized amplitude"
    )

    overlay_axis.set_xlim(
        0.0,
        LUNAR_DAY_HOURS,
    )

    overlay_axis.set_xticks(
        np.arange(0, 25, 2)
    )

    overlay_axis.grid(
        True,
        alpha=0.3,
    )

    overlay_axis.legend(
        ncol=2,
        fontsize=8,
    )

    # -------------------------------------------------------------
    # Pointwise product of all seven lunar days
    # -------------------------------------------------------------

    product_axis.plot(
        time_hours,
        product_all_days,
        color="black",
        linewidth=0.8,
        label="Product of all seven days",
    )

    product_axis.axhline(
        0.0,
        color="gray",
        linewidth=0.7,
        alpha=0.7,
    )

    product_axis.set_title(
        f"{label}: pointwise product of seven lunar days"
    )

    product_axis.set_ylabel(
        "Normalized amplitude product"
    )

    product_axis.set_xlim(
        0.0,
        LUNAR_DAY_HOURS,
    )

    product_axis.set_xticks(
        np.arange(0, 25, 2)
    )

    product_axis.grid(
        True,
        alpha=0.3,
    )

    product_axis.legend(
        fontsize=8,
    )


# Label the bottom row.
for axis in axes[-1, :]:
    axis.set_xlabel(
        "Time since beginning of lunar-day segment (hours)"
    )

fig.suptitle(
    f"AXBA1 and CZSHF lunar-day comparison: "
    f"{NUMBER_OF_LUNAR_DAYS} days",
    fontsize=15,
)

plt.show()