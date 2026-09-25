import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

sys.path.insert(0, os.path.abspath("."))
from scripts.run_part import run_part, get_default_db
from scripts.merge_parts import merge_parts

class HackathonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Amazon ML Hackathon - 1-Click Part Runner")
        self.root.geometry("680x560")
        self.root.resizable(False, False)

        # Style & Colors
        style = ttk.Style()
        style.theme_use("clam")

        # Header Frame
        header_frame = tk.Frame(root, bg="#1E293B", padx=20, pady=15)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame, 
            text="Amazon ML Hackathon - Model Runner", 
            font=("Segoe UI", 16, "bold"), 
            fg="#F8FAFC", 
            bg="#1E293B"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header_frame, 
            text="High-Precision Multi-Laptop Part Runner (Precision >= 99.5%)", 
            font=("Segoe UI", 9), 
            fg="#94A3B8", 
            bg="#1E293B"
        )
        sub_lbl.pack(anchor="w")

        # Main Container
        main_frame = tk.Frame(root, padx=25, pady=15, bg="#F1F5F9")
        main_frame.pack(fill="both", expand=True)

        # 1. Total Parts Selector
        part_box = tk.LabelFrame(main_frame, text=" 1. Part Selection ", font=("Segoe UI", 10, "bold"), bg="#F1F5F9", padx=15, pady=10)
        part_box.pack(fill="x", pady=6)

        tk.Label(part_box, text="Total Parts to divide dataset into:", bg="#F1F5F9").grid(row=0, column=0, sticky="w", pady=4)
        self.total_parts_var = tk.IntVar(value=4)
        total_parts_spin = ttk.Combobox(part_box, textvariable=self.total_parts_var, values=[2, 3, 4, 5, 8], width=8, state="readonly")
        total_parts_spin.grid(row=0, column=1, sticky="w", padx=10)
        total_parts_spin.bind("<<ComboboxSelected>>", self.update_part_buttons)

        tk.Label(part_box, text="Select Part for THIS laptop:", bg="#F1F5F9").grid(row=1, column=0, sticky="w", pady=8)
        self.part_var = tk.IntVar(value=1)

        self.btn_frame = tk.Frame(part_box, bg="#F1F5F9")
        self.btn_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=10)
        self.update_part_buttons()

        # 2. Paths
        paths_box = tk.LabelFrame(main_frame, text=" 2. File & Database Paths ", font=("Segoe UI", 10, "bold"), bg="#F1F5F9", padx=15, pady=8)
        paths_box.pack(fill="x", pady=6)

        default_input = "test_data/test_source1 (1).tsv" if os.path.exists("test_data/test_source1 (1).tsv") else "train_source1.tsv"
        tk.Label(paths_box, text="Input TSV:", bg="#F1F5F9").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar(value=default_input)
        tk.Entry(paths_box, textvariable=self.input_var, width=48).grid(row=0, column=1, padx=5, pady=3)

        tk.Label(paths_box, text="Database:", bg="#F1F5F9").grid(row=1, column=0, sticky="w")
        self.db_var = tk.StringVar(value=get_default_db())
        tk.Entry(paths_box, textvariable=self.db_var, width=48).grid(row=1, column=1, padx=5, pady=3)

        # 3. Action & Progress
        act_box = tk.LabelFrame(main_frame, text=" 3. Execution & Progress ", font=("Segoe UI", 10, "bold"), bg="#F1F5F9", padx=15, pady=10)
        act_box.pack(fill="both", expand=True, pady=6)

        self.status_lbl = tk.Label(act_box, text="Ready to run. Click 'START PROCESSING'.", fg="#0F172A", bg="#F1F5F9", font=("Segoe UI", 9))
        self.status_lbl.pack(anchor="w", pady=2)

        self.progress_bar = ttk.Progressbar(act_box, orient="horizontal", mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=6)

        btn_row = tk.Frame(act_box, bg="#F1F5F9")
        btn_row.pack(fill="x", pady=6)

        self.start_btn = tk.Button(
            btn_row, 
            text="▶ START PROCESSING THIS PART", 
            bg="#2563EB", 
            fg="white", 
            font=("Segoe UI", 10, "bold"),
            padx=15, 
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=5)

        self.merge_btn = tk.Button(
            btn_row, 
            text="🔗 MERGE ALL COMPLETED PARTS", 
            bg="#059669", 
            fg="white", 
            font=("Segoe UI", 10, "bold"),
            padx=15, 
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self.merge_all_parts
        )
        self.merge_btn.pack(side="right", padx=5)

    def update_part_buttons(self, event=None):
        for widget in self.btn_frame.winfo_children():
            widget.destroy()

        tot = self.total_parts_var.get()
        for p in range(1, tot + 1):
            rb = tk.Radiobutton(
                self.btn_frame, 
                text=f"Part {p}", 
                variable=self.part_var, 
                value=p,
                bg="#F1F5F9",
                font=("Segoe UI", 9, "bold" if p == 1 else "normal")
            )
            rb.pack(side="left", padx=4)

    def start_processing(self):
        part = self.part_var.get()
        tot = self.total_parts_var.get()
        inp = self.input_var.get().strip()
        db = self.db_var.get().strip()

        if not os.path.exists(inp):
            messagebox.showerror("File Error", f"Input file not found: {inp}")
            return
        if not os.path.exists(db):
            messagebox.showerror("DB Error", f"Database file not found: {db}\nTip: Run 'python scripts/build_test_db.py' first.")
            return

        self.start_btn.config(state="disabled")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start(10)
        self.status_lbl.config(text=f"Running Part {part} of {tot}... Check terminal for real-time progress.", fg="#2563EB")

        def worker():
            try:
                run_part(
                    input_file=inp,
                    db_path=db,
                    model_path="artifacts/models/matcher.joblib",
                    part_num=part,
                    total_parts=tot,
                    batch_size=1000
                )
                self.root.after(0, lambda: self.on_complete(part, tot))
            except Exception as e:
                self.root.after(0, lambda: self.on_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def on_complete(self, part, tot):
        self.progress_bar.stop()
        self.start_btn.config(state="normal")
        out_file = f"output/matching_results_part_{part}_of_{tot}.tsv"
        self.status_lbl.config(text=f"✓ Part {part} of {tot} Completed! Saved to: {out_file}", fg="#059669")
        messagebox.showinfo("Success", f"Part {part} of {tot} finished successfully!\nOutput saved to:\n{out_file}")

    def on_error(self, err_msg):
        self.progress_bar.stop()
        self.start_btn.config(state="normal")
        self.status_lbl.config(text=f"Error occurred: {err_msg[:60]}...", fg="#DC2626")
        messagebox.showerror("Execution Error", err_msg)

    def merge_all_parts(self):
        try:
            merge_parts(
                output_dir="output",
                input_source1_path=self.input_var.get().strip(),
                final_output_path="output/matching_results.tsv"
            )
            messagebox.showinfo("Merge Complete", "All parts successfully merged into:\noutput/matching_results.tsv\n\nFile is validated and ready for submission!")
        except Exception as e:
            messagebox.showerror("Merge Error", str(e))

def main():
    root = tk.Tk()
    app = HackathonApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
