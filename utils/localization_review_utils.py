"""Purpose: Persist human localization decisions and display sequential QC images.

Author: Qirui Zhang
Created: 2026-09-18
Adapted for the standalone library: 2026-09-27.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk


FAILURE_DECISIONS = ('T1_QC_fail', 'Hippo_Surgery', 'Other_QC_Fail')
PRIORITY_DECISIONS = ('Other_Surgery_lesion', 'T1_Enhanced')
DECISIONS = ('QC Pass', *FAILURE_DECISIONS, *PRIORITY_DECISIONS)


def write_csv_atomic(path, fields, rows):
    """Replace a table only after its complete content has been written."""
    staging = path.with_suffix('.csv.tmp')
    with staging.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    staging.replace(path)


class ReviewState:
    """Keep manual judgments separate from fixed automatic exclusion reasons."""

    def __init__(self, summary_path, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with Path(summary_path).open(encoding='utf-8', newline='') as stream:
            self.summary = list(csv.DictReader(stream))
        self.subjects = [row['subject'] for row in self.summary
                         if row['image_status'] == 'Images generated; visual review pending']
        self.manual_path = self.output_dir / 'manual_qc.csv'
        self.progress_path = self.output_dir / 'QC_fail_in_progress.csv'
        self.final_path = self.output_dir / 'QC_fail.csv'
        self.priority_path = self.output_dir / 'segmentation_priority_qc.csv'
        self.decisions = {}
        if self.manual_path.exists():
            with self.manual_path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['decision']:
                        if row['subject'] not in self.subjects or row['decision'] not in DECISIONS:
                            raise ValueError('Saved review does not match the current subject list or decision labels.')
                        self.decisions[row['subject']] = row
        if not self.complete and self.final_path.exists():
            raise RuntimeError('QC_fail.csv exists but the saved review is incomplete. Reconcile these files before reviewing.')

    @property
    def complete(self):
        return all(subject in self.decisions for subject in self.subjects)

    def first_pending(self):
        return next((i for i, subject in enumerate(self.subjects)
                     if subject not in self.decisions), None)

    def failure_rows(self):
        rows = []
        for subject_row in self.summary:
            subject = subject_row['subject']
            reasons = []
            if subject_row['image_status'] == 'Missing reference image':
                reasons.append('Missing_T1')
            if int(subject_row['n_valid_coordinates']) == 0:
                reasons.append('No_valid_corrected_coordinates')
            if subject in self.decisions:
                decision = self.decisions[subject]['decision']
                if decision in FAILURE_DECISIONS:
                    reasons.append(decision)
            if reasons:
                rows.append({'subject': subject, 'reason': '; '.join(reasons)})
        return rows

    def priority_rows(self):
        return [{'subject': subject, 'reason': self.decisions[subject]['decision']}
                for subject in self.subjects if subject in self.decisions
                and self.decisions[subject]['decision'] in PRIORITY_DECISIONS]

    def persist(self):
        records = [self.decisions.get(subject, {'subject': subject, 'decision': '', 'reviewed_at_utc': ''})
                   for subject in self.subjects]
        write_csv_atomic(self.manual_path, ['subject', 'decision', 'reviewed_at_utc'], records)
        failures = self.failure_rows()
        write_csv_atomic(self.progress_path, ['subject', 'reason'], failures)
        write_csv_atomic(self.priority_path, ['subject', 'reason'], self.priority_rows())
        if self.complete:
            write_csv_atomic(self.final_path, ['subject', 'reason'], failures)

    def record(self, subject, decision):
        if subject not in self.subjects or decision not in DECISIONS:
            raise ValueError('Unknown subject or review decision.')
        self.decisions[subject] = {
            'subject': subject, 'decision': decision,
            'reviewed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        self.persist()


class ReviewWindow:
    """Display six decisions, separating exclusions from segmentation priorities."""

    def __init__(self, window, state, image_dir):
        self.window = window
        self.state = state
        self.image_dir = Path(image_dir)
        self.index = state.first_pending()
        self.busy = False
        self.resize_job = None
        self.source_image = None
        self.photo = None
        window.title('Electrode Localization QC')
        window.geometry('1400x850')
        window.minsize(900, 600)
        if window.tk.call('tk', 'windowingsystem') == 'win32':
            window.state('zoomed')
        self.title = ttk.Label(window, font=('Arial', 13))
        self.title.pack(fill='x', padx=10, pady=(6, 2))
        self.canvas = tk.Canvas(window, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.queue_resize)
        buttons = ttk.Frame(window)
        buttons.pack(fill='x', padx=10, pady=8)
        self.buttons = []
        for column, decision in enumerate(DECISIONS):
            buttons.columnconfigure(column, weight=1)
            button = ttk.Button(buttons, text=decision, command=lambda value=decision: self.choose(value))
            button.grid(row=0, column=column, sticky='ew', padx=5, ipady=10)
            self.buttons.append(button)
        self.status = ttk.Label(window, font=('Arial', 10))
        self.status.pack(fill='x', padx=10, pady=(0, 6))
        window.bind('<Left>', lambda event: self.navigate(-1))
        window.bind('<Right>', lambda event: self.navigate(1))
        window.bind('<Home>', lambda event: self.go_pending())
        window.protocol('WM_DELETE_WINDOW', window.destroy)
        self.show_subject()

    def queue_resize(self, event=None):
        if self.resize_job is not None:
            self.window.after_cancel(self.resize_job)
        self.resize_job = self.window.after(80, self.draw_image)

    def draw_image(self):
        self.resize_job = None
        self.canvas.delete('all')
        if self.source_image is None:
            self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                    text='Review complete. QC_fail.csv is ready.\nPress Left to revisit decisions.',
                                    fill='white', font=('Arial', 22), justify='center')
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(width / self.source_image.width, height / self.source_image.height)
        size = (max(1, round(self.source_image.width * scale)), max(1, round(self.source_image.height * scale)))
        self.photo = ImageTk.PhotoImage(self.source_image.resize(size, Image.Resampling.LANCZOS))
        self.canvas.create_image(width / 2, height / 2, image=self.photo, anchor='center')

    def show_subject(self):
        self.busy = False
        reviewed = len(self.state.decisions)
        total = len(self.state.subjects)
        for button in self.buttons:
            button.config(state='normal' if self.index is not None else 'disabled')
        if self.index is None:
            self.source_image = None
            self.title.config(text=f'Completed: {reviewed}/{total} reviewed. {len(self.state.failure_rows())} excluded subjects. '
                              f'{len(self.state.priority_rows())} for priority segmentation QC.')
        else:
            subject = self.state.subjects[self.index]
            with Image.open(self.image_dir / f'{subject}.png') as image:
                self.source_image = image.convert('RGB')
            current = self.state.decisions.get(subject, {}).get('decision', 'Not reviewed')
            self.title.config(text=f'{subject}    Image {self.index + 1}/{total}    Reviewed {reviewed}/{total}    Decision: {current}')
        output = 'QC_fail.csv is final.' if self.state.complete else 'QC_fail_in_progress.csv is provisional; final export waits for all images.'
        self.status.config(text='Click a decision to save and advance. Left/Right: revisit. Home: first pending. ' + output)
        self.queue_resize()

    def choose(self, decision):
        if self.busy or self.index is None:
            return
        subject = self.state.subjects[self.index]
        try:
            self.state.record(subject, decision)
        except OSError as error:
            messagebox.showerror('Save failed', f'Cannot save the decision. Close files open in another application and retry.\n\n{error}')
            return
        self.busy = True
        for button in self.buttons:
            button.config(state='disabled')
        following = self.index + 1
        self.index = following if following < len(self.state.subjects) else self.state.first_pending()
        self.window.after(350, self.show_subject)  # Ignore a double click during the image transition.

    def navigate(self, step):
        if self.busy:
            return
        if self.index is None:
            self.index = len(self.state.subjects) - 1
        else:
            self.index = min(max(self.index + step, 0), len(self.state.subjects) - 1)
        self.show_subject()

    def go_pending(self):
        if not self.busy:
            self.index = self.state.first_pending()
            self.show_subject()
