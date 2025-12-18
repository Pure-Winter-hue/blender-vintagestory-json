# Winter's Development Branch

A slowly evolving “new” Blender tool, reflavored from the main branch.

**Status:** ⚠️ Highly Experimental

---

## Features

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

- **UV Unwrap: “View to Bounds”**
  - New unwrap mode designed for cuboids
  - Captures all **6 directions** of a model (press the 'Make Cuboid (rectify)' after, otherwise it auto-does this on export)
  - Speeds up UV layout work

---

## Notes

- This branch is under active development and may change rapidly.
- Expect rough edges, weirdness, and the occasional gremlin 🐛

---

## Contributing / Feedback

Issues, repro files, and screenshots are welcome.
If something breaks, include:
- Blender version
- Model JSON (or a minimal sample)
- Steps to reproduce
