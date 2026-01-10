import click
import cv2
from pathlib import Path
from .analysis import detect_color_mode, remove_tint, enhance_image
from .rotation import detect_skew_angle, rotate_image
from .trim import trim_image

@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
@click.option("--no-rotate", is_flag=True, help="Disable rotation correction")
@click.option("--no-trim", is_flag=True, help="Disable trimming")
@click.option("--no-tint", is_flag=True, help="Disable tint removal")
@click.option("--no-enhance", is_flag=True, help="Disable enhancement")
def main(files, output, no_rotate, no_trim, no_tint, no_enhance):
    for index, file in enumerate(files):

        path = Path(file)

        print(f"Processing: {path}")

        image = cv2.imread(str(path))

        # 1. Fix rotation
        if not no_rotate:
            angle = detect_skew_angle(image)
            print(f" - Detected Skew Angle: {angle:.2f} degrees")
            image = rotate_image(image, angle)

        # 2. Trim
        if not no_trim:
            image = trim_image(image)

        # 3. Remove tint
        if not no_tint:
            image = remove_tint(image)

        # 4. Analyze & Enhance
        if not no_enhance:
            mode = detect_color_mode(image)
            print(f" - Detected Mode: {mode}")
            image = enhance_image(image, mode)

        # 4. Save
        output_path = output.format(
            index=index,
            name=path.name,
            stem=path.stem,
        )
        cv2.imwrite(output_path, image)
        print(f"Success: Saved to {output_path}")

if __name__ == "__main__":
    main()