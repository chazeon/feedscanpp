import click
import cv2
from pathlib import Path
from .analysis import detect_color_mode, remove_tint, enhance_image
from .rotation import detect_skew_angle, rotate_image
from .trim import trim_image

@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
def main(files, output):
    for index, file in enumerate(files):

        path = Path(file)

        print(f"Processing: {path}")

        # 1. Fix rotation
        image = cv2.imread(str(path))
        angle = detect_skew_angle(image)
        image = rotate_image(image, angle)

        # 2. Trim
        image = trim_image(image)

        # 3. Analyze & Enhance
        image = remove_tint(image)

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