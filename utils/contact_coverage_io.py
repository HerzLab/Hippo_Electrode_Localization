"""Purpose: Write contact coverage GIFTI metrics and the plain fixed-input contact workbook.

Author: Qirui Zhang
Created: 2026-09-21
Adapted for the standalone library: 2026-09-27.
"""

import nibabel as nib
import xlsxwriter


def save_surface_metric(path, values, metadata):
    """Write one scalar array in the original midthickness vertex order."""
    array = nib.gifti.GiftiDataArray(data=values, intent="NIFTI_INTENT_SHAPE")
    image = nib.gifti.GiftiImage(darrays=[array], meta=nib.gifti.GiftiMetaData(metadata))
    nib.save(image, path)


def save_contact_table(path, rows, columns, subject_rows, subject_columns):
    """Write the fixed Contacts and subjects sheets without fills or formulas."""
    with xlsxwriter.Workbook(path, {"strings_to_formulas": False, "strings_to_urls": False}) as workbook:
        text_style = workbook.add_format({"font_name": "Arial", "font_size": 10})
        number_style = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000000"})
        heading_style = workbook.add_format({"font_name": "Arial", "font_size": 10, "bold": True, "text_wrap": True})
        for name, sheet_rows, sheet_columns in [("Contacts", rows, columns), ("subjects", subject_rows, subject_columns)]:
            sheet = workbook.add_worksheet(name)
            sheet.set_default_row(15)
            sheet.set_row(0, 45)
            sheet.write_row(0, 0, sheet_columns, heading_style)
            for column_index, column in enumerate(sheet_columns):
                width = max(20, len(column) + 2)  # Keep machine-readable field names fully visible.
                sheet.set_column(column_index, column_index, width, text_style)
            for row_index, row in enumerate(sheet_rows, start=1):
                for column_index, column in enumerate(sheet_columns):
                    value = row[column]
                    style = number_style if isinstance(value, float) else text_style
                    sheet.write(row_index, column_index, value, style)
            sheet.freeze_panes(1, 2 if name == "Contacts" else 1)
            sheet.autofilter(0, 0, len(sheet_rows), len(sheet_columns) - 1)
