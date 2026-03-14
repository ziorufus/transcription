from pathlib import Path
import subprocess
import yaml


def extract_wave_portions(wav_file: str, segments: list, output_folder: str, delta: float = 0.1) -> float:
    """
    Extract portions of a WAV file based on entries in a YAML file.

    Parameters
    ----------
    wav_file : str
        Path to the input WAV file.
    segments : list
        List of segment dictionaries containing 'offset' and 'duration' keys.
    output_folder : str
        Folder where extracted portions will be saved.
    delta : float, optional
        Delta to add to offset and duration (default: 0.1).

    YAML format example
    -------------------
    [
      {duration: 15.9359, offset: 2.042, wav: audio.wav},
      {duration: 2.4424, offset: 19.5596, wav: audio.wav}
    ]

    Output files
    ------------
    portion_0.wav, portion_1.wav, ...
    """
    wav_path = Path(wav_file)
    out_dir = Path(output_folder)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(segments, list):
        raise ValueError("Segments must be a list of dictionaries.")

    total_duration = 0.0

    for i, segment in enumerate(segments):
        try:
            offset = float(segment["offset"] - delta)
            duration = float(segment["duration"] + 2 * delta)
        except KeyError as e:
            raise ValueError(f"Missing required key in segment {i}: {e}")
        except (TypeError, ValueError):
            raise ValueError(f"Invalid offset/duration in segment {i}: {segment}")

        total_duration += segment["duration"]
        output_file = out_dir / f"portion_{i}.wav"

        cmd = [
            "ffmpeg",
            "-y",                    # overwrite output if it exists
            "-i", str(wav_path),     # input file
            "-ss", str(offset),      # start time
            "-t", str(duration),     # duration
            "-c", "copy",            # no re-encoding
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for segment {i}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error:\n{result.stderr}"
            )
    
    return total_duration