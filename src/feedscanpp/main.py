import click
import cv2
from pathlib import Path
from .detector import DocumentDetector
from .transform import four_point_transform
from .analysis import detect_color_mode, remove_tint

@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--debug/--no-debug", default=False, help="Save debug images with detected corners")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
def main(files, output, debug=False):

    detector = DocumentDetector()

    for index, file in enumerate(files):

        path = Path(file)

        print(f"Processing: {path}")

        # 1. Detect Document Layout
        try:
            corners = detector.find_corners(file)
            print(f" - Detected corners: {corners}")
        except Exception as e:
            print(f"Detection Error: {e}")
            return

        image = cv2.imread(file)

        if debug:
            image_debug = image.copy()
            for point in corners:
                if point is not None:
                    center = (int(point[0]), int(point[1]))
                    debug = cv2.circle(image_debug, center, 25, (0, 0, 255), -1)
                cv2.imwrite(f"{path.stem}.debug.png", image_debug)  # Debugging

        image = four_point_transform(image, corners)

        print(" - Document cropped successfully.")

        # 3. Analyze & Enhance
        image = remove_tint(image)

        mode = detect_color_mode(image)
        print(f" - Detected Mode: {mode}")

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