# Winter's Development Branch
<img width="1920" height="1080" alt="vsblendertool" src="https://github.com/user-attachments/assets/8f26c697-6284-4593-ab34-a5d5a88a59f9" />

A slowly evolving “new” Blender tool, reflavored from the main branch.
Credits: Phonon worked massively hard to make the original, Sekelsta for identifying some blender to VS comptibility workflow and providing sample files to test on, everyone who contributed to the main plugin which is the backbone of this extension of it.

**Status:** ⚠️ Semi Experimental 

---

## Features
<img width="276" height="907" alt="image" src="https://github.com/user-attachments/assets/f1b5f401-ddb1-4b15-8fa1-62e11669edde" />


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
  - Shortest distance rotation <- use first. - (In truly broken cases use with the 'sanitation' button (experimental baking may not be useful) for any 'it looks fine in vsmc but not in game!!' scnearios. :) 
<img width="336" height="270" alt="image" src="https://github.com/user-attachments/assets/d528bf2a-be37-4499-aa80-8dd8503cc73f" />


- **UV Unwrap: “Directional Entity Unwrap”**
  - Pure Winter's favored UV method, but now automated with a blender tool! Creates a wonderful layout for 2D texture artists to draw directly on, with good flow from one body piece to another. 
  - This is designed to go from the tip of the nose, down the back of the neck, over the back, to the tail tip. (Sections of 4-6 pieces generally unwrap in the correct order, if it doesn't, shorten yourselection. If it unwraps horizontal instead of vertical, simply rotate that model part and unwrap again. Uses your viewport to decide what is right or left, so you never have to worry about what side is really up/down/east/west while modeling.
  -After the back, you are meant to do the outside of the legs, then switch to 'single direction unwrap' to do the inside of the legs, and the belly.
  -Don't forget to test on a texture background so you know if you need to mirror a uv or not- feel free to also stack the uv's. :D
<img src=https://i.imgur.com/5gciwIL.gif>

- **UV Unwrap: “Single Direction Unwrap”**
-Identical to Directional Entity Unwrap, but only unwraps the faces you selected. Afterall if you already did the front of the leg, outer leg, and back of the leg- you now only need to do the inside.
-This keeps it lined up and proportionate identically to the directional entity unwrap so you can align your UV's left to right on the same 'level of the body'. I.e. if you paint a bar left to right, it will make a perfectly straight line all the way aorund the leg.
-Additionally, if you select only one face, and click the button, it will unwrap it in the oreintation of your viewport which makes for quick click and drag layouts for odd spots.

- **UV Unwrap: “View to Bounds”**
  - New unwrap mode designed for cuboids
  - Captures all **6 directions** of a model (press the 'Make Cuboid (rectify)' after, otherwise it auto-does this on export)
  - Speeds up UV layout work
  <img width="1673" height="591" alt="image" src="https://github.com/user-attachments/assets/0bfd1b73-b74b-4822-9132-98d52aee36a4" />
  <img width="1723" height="651" alt="image" src="https://github.com/user-attachments/assets/0a676a86-0976-4b5f-9a05-a8245e99c26d" />



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
