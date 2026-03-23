// ============================================================================
// FIJI MACRO: Apply Local Thickness to segmented images
// ============================================================================

// Input and output directories
// *** SET THESE PATHS TO YOUR LOCAL FOLDERS BEFORE RUNNING ***
inputDir = "/path/to/your/005_membranes/";
outputDir = "/path/to/your/007_local_thickness_results/";

// Create output directory if it doesn't exist
File.makeDirectory(outputDir);

// Get list of files in input directory
list = getFileList(inputDir);

// Counter for processed files
count = 0;

// Process each file
setBatchMode(true);  // Run in batch mode for speed

for (i = 0; i < list.length; i++) {
    filename = list[i];
    
    // Check if file is a TIF/TIFF
    if (endsWith(filename, ".tif") || endsWith(filename, ".tiff")) {
        // Open the image
        open(inputDir + filename);
        
        // Get the image title
        imageTitle = getTitle();
        
        // Run Local Thickness with threshold=2 parameter
        run("Local Thickness (complete process)", "threshold=2");
        
        // The result image will be opened automatically
        // Get the result title
        resultTitle = getTitle();
        
        // Save the result
        outputFilename = replace(filename, ".tif", "_LocThk.tif");
        saveAs("Tiff", outputDir + outputFilename);
        
        count++;
        print("Processed: " + filename + " -> " + outputFilename + " (" + count + "/" + list.length + ")");
        
        // Close all images
        close("*");
    }
}

setBatchMode(false);

// Print summary
print("========================================");
print("Local Thickness processing complete!");
print("Total files processed: " + count);
print("Output saved to: " + outputDir);
print("========================================");