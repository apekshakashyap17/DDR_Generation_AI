import fitz
import os

def convert_pdf_to_images(pdf_path, output_folder, prefix, dpi=150):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    print(f"Opening {pdf_path}")
    print(f"Total pages found: {len(doc)}")
    print(f"Converting pages to images at {dpi} DPI...")

    saved_images = []

    for page_number in range(len(doc)):

        # Loading the page
        page = doc[page_number]

        # Rendering page to pixel map
        pixmap = page.get_pixmap(matrix=matrix)

        # Build the output file name
        image_name = f"{prefix}_page_{str(page_number + 1).zfill(2)}.jpg"
        image_path = os.path.join(output_folder, image_name)

        # Save the pixel map as JPEG
        pixmap.save(image_path)

        saved_images.append(image_path)

        print(f"Saved: {image_name}")

    doc.close()

    print(f"Done. {len(saved_images)} images saved to {output_folder}")
    print("")

    return saved_images

if __name__ == "__main__":

    # input PDF paths
    inspection_pdf = r"D:\UrbanRoof assignment\sample data\Sample Report.pdf"
    thermal_pdf    = r"D:\UrbanRoof assignment\sample data\Thermal Images.pdf"

    # output folder for images
    pages_folder = "pages"

    # Convert inspection report
    inspection_images = convert_pdf_to_images(
        pdf_path      = inspection_pdf,
        output_folder = pages_folder,
        prefix        = "inspection"
    )

    # Convert thermal report
    thermal_images = convert_pdf_to_images(
        pdf_path      = thermal_pdf,
        output_folder = pages_folder,
        prefix        = "thermal"
    )

    print("Both PDFs converted successfully.")
    print(f"Total images ready: {len(inspection_images) + len(thermal_images)}")
