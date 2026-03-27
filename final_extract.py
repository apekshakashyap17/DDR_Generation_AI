import os
import base64
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Load API Key
# ----------------------------

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Please check your .env file."
    )

# Initialize OpenRouter client
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key  = api_key
)

# Model to use
MODEL = "openai/gpt-4.1-mini"


# ----------------------------
# Load Images From Pages Folder
# ----------------------------

def load_images_from_folder(folder, prefix):
    """
    Loads all images for a given prefix
    (inspection or thermal) from the pages folder,
    sorted by page number.
    """

    folder_path = Path(folder)

    image_files = sorted(
        [f for f in folder_path.glob(f"{prefix}_page_*.jpg")]
    )

    if not image_files:
        raise FileNotFoundError(
            f"No images found for prefix '{prefix}' "
            f"in folder '{folder}'"
        )

    images = []

    for image_file in image_files:
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Convert to base64 for OpenRouter
        b64_data = base64.b64encode(image_data).decode("utf-8")

        images.append({
            "filename" : image_file.name,
            "b64"      : b64_data,
            "mime_type": "image/jpeg"
        })

    print(f"Loaded {len(images)} {prefix} images")
    return images


# ----------------------------
# Build Image Messages
# ----------------------------

def build_image_messages(images, instruction):
    """
    Builds the message content list in OpenRouter
    format with text and images combined.
    """

    content = []

    # Add the instruction text first
    content.append({
        "type": "text",
        "text": instruction
    })

    # Add each image with its filename label
    for img in images:
        # Add filename label
        content.append({
            "type": "text",
            "text": f"Page: {img['filename']}"
        })

        # Add the image
        content.append({
            "type"     : "image_url",
            "image_url": {
                "url": f"data:{img['mime_type']};base64,{img['b64']}"
            }
        })

    return content


# ----------------------------
# Send Batch to OpenRouter
# ----------------------------

def send_batch(images, instruction, batch_label=""):
    """
    Sends a single batch of images to OpenRouter
    with a specific instruction.
    Returns the text response.
    """

    print(f"Sending {batch_label} to OpenRouter...")

    content = build_image_messages(images, instruction)

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role"   : "user",
                "content": content
            }
        ],
        max_tokens = 4096
    )

    result = response.choices[0].message.content

    print(f"{batch_label} done. "
          f"Received {len(result)} characters.")

    return result


# ----------------------------
# DDR Generation Prompt
# ----------------------------

DDR_PROMPT = DDR_PROMPT = """
You are a senior property inspection engineer with 
20 years of experience writing Detailed Diagnostic 
Reports for residential clients.

You have been given two sets of extracted findings:

DOCUMENT 1 - INSPECTION REPORT FINDINGS
Site observations, area-wise issues, checklist results
with severity indicators, photo references, and a
summary table mapping causes to affected areas.

DOCUMENT 2 - THERMAL IMAGING FINDINGS  
Thermal camera readings with hotspot and coldspot
temperatures, visible photo descriptions, area
matching results, and moisture risk assessments.

YOUR TASK:
Combine both sets of findings into one complete,
accurate, client-friendly DDR report.

STRICT RULES — FOLLOW THESE EXACTLY:
1. Do NOT invent any facts not present in the findings
2. Do NOT use technical jargon the client cannot understand
3. If information is missing write exactly: Not Available
4. If inspection and thermal findings conflict,
   write: CONFLICT NOTED — then describe both sides
5. Do not repeat the same observation twice
6. Every area mentioned in the inspection report
   MUST appear in Section 2
7. Every thermal scan MUST be referenced somewhere
   in the report even if the area match is unclear
8. Severity ratings MUST follow these exact definitions:
   CRITICAL = Active plumbing failure confirmed OR
              structural damage affecting load bearing
              elements OR issue affecting more than
              3 rooms simultaneously
   HIGH     = Confirmed ongoing moisture with visible
              damage in one or two areas OR active
              leakage confirmed by thermal AND visual
   MODERATE = Moisture present but contained OR
              visible staining without confirmed
              active source
   LOW      = Historic staining only with no current
              active moisture confirmed by thermal

REPORT STRUCTURE:
Generate the report in exactly this format.
Do not add extra sections or rename any section.

---

DETAILED DIAGNOSTIC REPORT (DDR)

PROPERTY DETAILS
- Property Type:
- Number of Floors:
- Inspection Date:
- Inspected By:
- Overall Inspection Score:
- Previous Structural Audit:
- Previous Repair Work Done:

---

1. PROPERTY ISSUE SUMMARY

Write 2 to 3 paragraphs covering:
- Overall condition of the property in simple terms
- Total number of distinct issues found
- Which areas are affected
- General severity level across the property
- Whether issues appear active or historical
- One sentence on urgency of action required

Write this for a non-technical property owner.
Avoid all engineering jargon.

---

2. AREA-WISE OBSERVATIONS

For EVERY impacted area found in the inspection
report write the following. Do not skip any area.

AREA: [Area Name and Flat Number]

Problem Location (Negative Side):
[Describe exactly where the problem is visible,
which wall, which level, what it looks like]

Source of Problem (Positive Side):
[Describe exactly where the cause is coming from,
which bathroom, which fixture, which crack,
and explain in simple terms how the source
is causing the problem at the negative side]

Thermal Evidence:
[Write the exact thermal image filename such as
RB02380X.JPG. Then write the exact coldspot
temperature, exact hotspot temperature, and
calculate the exact temperature delta.
State the moisture risk level as LOW, MODERATE,
or HIGH based on the delta value.
Describe in one sentence what the thermal image
shows at this location.
Do NOT write page ranges.
Cite one specific thermal image per area minimum.
If no thermal image could be matched write:
Thermal image Not Available — explain why briefly]

Photo Evidence:
[List every individual photo number that shows
this area. Write each photo number separately.
For example write: Photo 1, Photo 3, Photo 5, Photo 7
Do NOT write ranges like Photos 1 to 7.
Do NOT write page numbers instead of photo numbers.
If no photos exist for this area write:
Photo evidence Not Available]

---

3. PROBABLE ROOT CAUSE

For the property overall and for each area
individually explain in plain simple language:
- What is the single most likely cause
- Why it has spread to multiple areas
- Whether it is a construction defect,
  maintenance failure, or age related issue
- How long the issue has likely been present
  based on visible damage severity

---

4. SEVERITY ASSESSMENT

For each area provide:

[Area Name]: [CRITICAL / HIGH / MODERATE / LOW]
Reasoning: [Explain exactly why this rating was
given using the severity definitions. Reference
specific thermal readings and visual observations
that support the rating. Do not give a rating
without reasoning.]

Overall Property Severity: [Single rating]
Overall Reasoning: [One paragraph]

---

5. RECOMMENDED ACTIONS

List actions in priority order from most urgent
to least urgent. For each action write:

Priority [Number] — [Action Title]
Area affected: [which areas this fixes]
What to do: [specific steps in plain language]
Why urgent: [consequence if not done]
Estimated complexity: [Simple / Moderate / Complex]

---

6. ADDITIONAL NOTES

Include observations that do not fit above sections:
- Building wide patterns observed
- Maintenance quality observations  
- Risk of future issues if current problems
  are left unaddressed
- Any positive findings worth noting
- Recommendations for ongoing monitoring

---

7. MISSING OR UNCLEAR INFORMATION

List any information that was expected but not found 
in the documents. For each missing item write:
- What is missing
- Why it matters
- Write "Not Available" next to each item

Also note any conflicts found between inspection
and thermal data under:

CONFLICTS DETECTED:
[List any contradictions or write:
No conflicts detected]
---

END OF REPORT

---

Important final check before submitting your response:
- Have you covered every impacted area from the
  inspection report in Section 2?
- Have you referenced every thermal scan somewhere?
- Does every severity rating have reasoning?
- Is every recommended action in priority order?
- Have you written Not Available for everything missing?

If any of these checks fail, fix the report before
responding.
"""


# ----------------------------
# Extraction Instructions
# ----------------------------

INSPECTION_INSTRUCTION = INSPECTION_INSTRUCTION = """
You are reading pages from a professional property 
inspection report. Your job is to extract every piece 
of information with complete accuracy.

Extract the following in structured text:

PROPERTY DETAILS:
- Property Type
- Number of Floors
- Inspection Date and Time
- Inspected By
- Overall Score
- Previous Structural Audit Done (Yes/No)
- Previous Repair Work Done (Yes/No)

FLAGGED ITEMS:
- List every flagged checklist item
- State exactly what was flagged and why

CHECKLIST FINDINGS:
For every single checklist row extract:
- The exact question text
- The answer given
- The severity level using these exact labels:
  RED    = CRITICAL ISSUE (flagged)
  AMBER  = WARNING (needs attention)
  BROWN  = NOT APPLICABLE
  GREEN  = ACCEPTABLE
- Whether this item was flagged or not

IMPACTED AREAS:
For every impacted area section extract:
- Area name and flat number
- Negative side description
  (exactly where the problem is visible,
   which wall, which level, which room)
- Positive side description
  (exactly where the source or cause is,
   which bathroom, which fixture, which crack)
- All photo numbers listed for this area
  (e.g. Photo 1 to Photo 7)
- Page number where this area appears

SUMMARY TABLE:
Extract the complete summary table if present:
- Every point number
- Impacted area description on negative side
- Exposed area description on positive side
- The relationship between negative and positive
  (explain in one sentence what is causing what)

STRUCTURAL OBSERVATIONS:
- Condition of RCC columns and beams
- Any cracks observed (location and severity)
- Any corrosion or spalling observed
- External wall condition
- Paint condition and type if mentioned

Extract everything. Do not summarize or skip any detail.
If a page appears blank or has no relevant content
write: Page appears to contain no extractable data.
"""

THERMAL_INSTRUCTION = THERMAL_INSTRUCTION = """
You are reading pages from a thermal imaging report
taken during a property inspection.

Each page follows the same layout:
- Top left: thermal image showing heat distribution
- Top right: metadata table with temperature readings
- Bottom: visible light photograph of the same spot

For every single page extract:

THERMAL IMAGE ID:
- Exact filename shown (e.g. RB02380X.JPG)
- Page number in this document

TEMPERATURE READINGS:
- Hotspot temperature (exact value with unit)
- Coldspot temperature (exact value with unit)
- Reflected temperature (exact value with unit)
- Emissivity value
- Date of scan

THERMAL ANALYSIS:
- Temperature delta (hotspot minus coldspot)
- Location of the coldspot in the thermal image
  (describe precisely: bottom left corner,
   skirting level, ceiling edge, wall base etc.)
- Color zones visible in thermal image
  (blue/cyan = cold = moisture,
   green = ambient,
   red/orange = warm)
- Severity of cold zone:
  Delta less than 2°C  = LOW moisture risk
  Delta 2 to 4°C       = MODERATE moisture risk
  Delta more than 4°C  = HIGH moisture risk

VISIBLE PHOTO DESCRIPTION:
- What is physically visible in the photo below
  the thermal image
- Describe the room or area shown
- Note any visible dampness, staining, cracks,
  or damage visible in the photo
- Try to identify which room this is based on
  visible features (tiles, fixtures, walls, windows)

AREA MATCHING:
- Based on the visible photo, which room or area
  does this thermal scan most likely correspond to?
- State your confidence: CERTAIN / PROBABLE / UNCLEAR
- Explain your reasoning in one sentence

Extract every page completely.
Do not skip any page even if it looks similar
to previous pages.
"""


# ----------------------------
# Main DDR Generation Function
# ----------------------------

def generate_ddr(pages_folder="pages"):
    """
    Loads all inspection and thermal images,
    sends them to OpenRouter in small batches,
    combines extracted text, then generates
    the full DDR report.
    """

    print("Loading inspection images...")
    inspection_images = load_images_from_folder(
        pages_folder, "inspection"
    )

    print("Loading thermal images...")
    thermal_images = load_images_from_folder(
        pages_folder, "thermal"
    )

    # ----------------------------------------
    # Split into batches of 8 pages each
    # ----------------------------------------

    def chunk_list(lst, chunk_size):
        return [
            lst[i:i+chunk_size]
            for i in range(0, len(lst), chunk_size)
        ]

    inspection_batches = chunk_list(inspection_images, 8)
    thermal_batches    = chunk_list(thermal_images, 8)

    print(f"\nInspection split into "
          f"{len(inspection_batches)} batches")
    print(f"Thermal split into "
          f"{len(thermal_batches)} batches")

    # ----------------------------------------
    # Process inspection batches
    # ----------------------------------------

    inspection_results = []

    for i, batch in enumerate(inspection_batches):
        label = (f"Inspection batch {i+1} of "
                 f"{len(inspection_batches)} "
                 f"({batch[0]['filename']} to "
                 f"{batch[-1]['filename']})")

        print(f"\nProcessing {label}...")

        result = send_batch(
            batch,
            INSPECTION_INSTRUCTION,
            label
        )
        inspection_results.append(result)

        # Wait between batches
        if i < len(inspection_batches) - 1:
            print("Waiting 10 seconds before next batch...")
            time.sleep(10)

    # ----------------------------------------
    # Process thermal batches
    # ----------------------------------------

    thermal_results = []

    for i, batch in enumerate(thermal_batches):
        label = (f"Thermal batch {i+1} of "
                 f"{len(thermal_batches)} "
                 f"({batch[0]['filename']} to "
                 f"{batch[-1]['filename']})")

        print(f"\nProcessing {label}...")

        result = send_batch(
            batch,
            THERMAL_INSTRUCTION,
            label
        )
        thermal_results.append(result)

        # Wait between batches
        if i < len(thermal_batches) - 1:
            print("Waiting 10 seconds before next batch...")
            time.sleep(10)

    # ----------------------------------------
    # Combine all extracted text
    # ----------------------------------------

    print("\nCombining all extracted findings...")

    combined = "INSPECTION REPORT FINDINGS:\n\n"
    for i, text in enumerate(inspection_results):
        combined += f"--- Inspection Part {i+1} ---\n"
        combined += text + "\n\n"

    combined += "\nTHERMAL IMAGING FINDINGS:\n\n"
    for i, text in enumerate(thermal_results):
        combined += f"--- Thermal Part {i+1} ---\n"
        combined += text + "\n\n"

    # ----------------------------------------
    # Generate final DDR report
    # ----------------------------------------

    print("\nGenerating final DDR report...")
    print("This is the last API call, please wait...")
    time.sleep(5)

    final_response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role"   : "user",
                "content": combined + "\n\n" + DDR_PROMPT
            }
        ],
        max_tokens = 8192
    )

    ddr_text = final_response.choices[0].message.content

    print("\nDDR report generated successfully.")
    print(f"Report length: {len(ddr_text)} characters")

    return ddr_text


# ----------------------------
# Save DDR Text to File
# ----------------------------

def save_ddr_text(ddr_text, output_folder="output"):
    """
    Saves the raw DDR text to a file so you can
    inspect it before building the Word document.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "ddr_raw.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ddr_text)

    print(f"\nRaw DDR text saved to {output_path}")
    return output_path


# ----------------------------
# Run this file directly
# ----------------------------

if __name__ == "__main__":

    # Generate the DDR
    ddr_text = generate_ddr(pages_folder="pages")

    # Save raw text output
    save_ddr_text(ddr_text, output_folder="output")

    print("\nDDR generation complete.")
    print("Check output/ddr_raw.txt to review the report.")
