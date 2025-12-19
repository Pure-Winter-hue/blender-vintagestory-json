# Winter's Development Branch
<img width="1920" height="1080" alt="vsblendertool" src="https://github.com/user-attachments/assets/8f26c697-6284-4593-ab34-a5d5a88a59f9" />

A slowly evolving “new” Blender tool, reflavored from the main branch.
Credits: Phonon worked massively hard to make the original, Sekelsta for identifying some blender to VS comptibility workflow and providing sample files to test on, everyone who contributed to the main plugin which is the backbone of this extension of it.

**Status:** ⚠️ Semi Experimental 

---

## Features
<img width="277" height="810" alt="image" src="https://github.com/user-attachments/assets/13769071-166e-40c2-8fab-23bd43f0e1db" />

 - **Disabled Faces Support**
   - Maintains VSMC disabled faces on import/export. (Yes, even hair/feather cards.)
   - You can now disable faces for VSMC from the tool. (press N to see tool.) Go to edit mode, face selection mode, click face, click button on tool. [This is different then deleting or hiding a face in blender.]
   <img width="953" height="423" alt="Screenshot 2025-12-19 065909" src="https://github.com/user-attachments/assets/772f90af-1e36-4eed-8c2b-e101b9d1c96d" />


- **Mirror Button**
  - Duplicates selection (includes disabled faces/armature!)
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
  - Shortest distance rotation - use with the 'sanitation' button (baking may not be useful) for any 'it looks fine in vsmc but not in game!!' scnearios. :) 
  <img width="709" height="283" alt="image" src="https://github.com/user-attachments/assets/c74971db-4bf3-47be-960f-d18338993f78" />


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
