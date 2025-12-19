# Winter's Development Branch
<img width="1920" height="1080" alt="vsblendertool" src="https://github.com/user-attachments/assets/8f26c697-6284-4593-ab34-a5d5a88a59f9" />

A slowly evolving “new” Blender tool, reflavored from the main branch.
Credits: Phonon worked massively hard to make the original, Sekelsta for identifying some blender to VS comptibility workflow and providing sample files to test on, everyone who contributed to the main plugin which is the backbone of this extension of it.

**Status:** ⚠️ Semi Experimental 

---

## Features
<img width="277" height="810" alt="image" src="https://github.com/user-attachments/assets/13769071-166e-40c2-8fab-23bd43f0e1db" />


- **Mirror Button**
  - Duplicates selection
  - Auto-renames `L`/`Left` → `R`/`Right`
  - Flips geometry and bones relative to the **3D Cursor**
    - Tip: place the cursor at **World Origin** for the intended workflow

- **Single Material Workflow**
  - Removes all materials and assigns a single default material: **`skin`**
  - Prevents `null` materials

- **Clean JSON Names (No Hidden Decimals)**
  - Renames mesh data to match object names
  - Runs on **import and export**
  - Helps avoid Blender-style suffixes like `.001` ending up in exported JSON

- **Animation Import/Export**
  - Imports and exports Vintage Story animations with minimal fuss
  - Shortest distance rotation + baking (use both together for best results) for STUBBORN older blender animations. <- Winter version, not from base tool.

- **UV Unwrap: “View to Bounds”**
  - New unwrap mode designed for cuboids
  - Captures all **6 directions** of a model (press the 'Make Cuboid (rectify)' after, otherwise it auto-does this on export)
  - Speeds up UV layout work

---

## Notes

- This branch is under active development and may change rapidly.
- Expect rough edges, weirdness, and the occasional gremlin 🐛

---

## Contributing / Feedback - 

Issues, repro files, and screenshots are welcome in 'issues' or directly on discord in its dedicated channel:
https://discord.com/channels/302152934249070593/1451452685520998440/1451452685520998440
If something breaks, include:
- Blender version
- Model JSON (or a minimal sample)
- Steps to reproduce
