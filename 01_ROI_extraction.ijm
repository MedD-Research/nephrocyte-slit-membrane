// ==========================
// ROI + Mask export macro
// Saves per ROI:
//   1) "Original_image"_ROIxxx.tif
//   2) "Original_image"_ROIxxx_mask.tif
// ==========================

// ---- Set output folders here (or leave "" to be asked once) ----
// *** SET THESE PATHS TO YOUR LOCAL FOLDERS BEFORE RUNNING ***
// Leave as "" to be prompted with a folder dialog when the macro runs
roiDir  = "";  // e.g. "/Users/you/project/001_rois/"
maskDir = "";  // e.g. "/Users/you/project/001_masks/"

// Get the active image
original = getImageID();
originalName = getTitle();

// Remove extension from original name
dotIndex = lastIndexOf(originalName, ".");
if (dotIndex > 0) originalName = substring(originalName, 0, dotIndex);

// Ask for folders if not set or missing
if (roiDir == "" || !File.exists(roiDir))   roiDir  = getDirectory("Folder to save ROIs");
if (maskDir == "" || !File.exists(maskDir)) maskDir = getDirectory("Folder to save masks");

// Ensure trailing separators
if (!endsWith(roiDir, File.separator))  roiDir  = roiDir  + File.separator;
if (!endsWith(maskDir, File.separator)) maskDir = maskDir + File.separator;

// Create folders if needed (getDirectory returns existing, but if user typed path into roiDir/maskDir)
if (!File.exists(roiDir))  File.makeDirectory(roiDir);
if (!File.exists(maskDir)) File.makeDirectory(maskDir);

print("Saving ROIs to:  " + roiDir);
print("Saving masks to: " + maskDir);

// Ensure background is black (important for Clear Outside)
setBackgroundColor(0, 0, 0);

roiCount = roiManager("count");
if (roiCount == 0) exit("ROI Manager is empty.");

for (i = 0; i < roiCount; i++) {
    selectImage(original);
    roiManager("select", i);

    // ROI index as 3 digits: 000, 001, 002...
    roiIndex = d2s(i, 0);
    if (i < 10) roiIndex = "00" + roiIndex;
    else if (i < 100) roiIndex = "0" + roiIndex;

    roiFilename  = roiDir  + originalName + "-ROI-" + roiIndex + ".tif";
    maskFilename = maskDir + originalName + "-ROI-" + roiIndex + "_mask.tif";

    // Skip if both already exist
    if (File.exists(roiFilename) && File.exists(maskFilename)) {
        print("Skipping ROI " + roiIndex + " - files already exist");
        continue;
    }

    // Duplicate ROI region (bounding-box crop); ROI selection is preserved and shifted correctly
    run("Duplicate...", "title=ROI_data_" + roiIndex);
    dataID = getImageID();

    // Keep only ROI content (corners/background become 0)
    run("Clear Outside");

    // Optional: keep consistent bit depth for saving
    // (Comment out if you want to preserve original bit depth)
    if (bitDepth != 8) run("8-bit");

    // Create mask from the ROI selection on the CROPPED image (same canvas)
    run("Create Mask");
    maskID = getImageID();
    rename("ROI_mask_" + roiIndex);

    if (bitDepth != 8) run("8-bit");

    // Save ROI image and mask image separately
    selectImage(dataID);
    saveAs("Tiff", roiFilename);

    selectImage(maskID);
    saveAs("Tiff", maskFilename);

    // Close intermediates
    selectImage(dataID); close();
    selectImage(maskID); close();

    print("Saved ROI " + roiIndex);
}

selectImage(original);
print("Done! Saved " + roiCount + " ROIs and masks.");
