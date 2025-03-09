import React, { useState, useEffect } from "react";
import { ArrowUpTrayIcon, ArrowDownTrayIcon } from "@heroicons/react/24/outline";

const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 500, height: 300 });
  const [colorizedImage, setColorizedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.[0]) return;

    const file = event.target.files[0];
    setSelectedFile(file);
    setColorizedImage(null); // Reset colorized image if a new file is selected

    const imageUrl = URL.createObjectURL(file);
    setPreviewImage(imageUrl);

    // Create an image element to get its original dimensions
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const { width, height } = img;

      // Scale the image while keeping aspect ratio
      const maxSize = 512;
      let newWidth = width;
      let newHeight = height;

      if (width > height) {
        newWidth = maxSize;
        newHeight = (height / width) * maxSize;
      } else {
        newHeight = maxSize;
        newWidth = (width / height) * maxSize;
      }

      // Add 20 to each dimension for a 10px border
      setImageSize({
        width: Math.round(newWidth) + 20,
        height: Math.round(newHeight) + 20,
      });
    };
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    console.log("Uploading file:", selectedFile.name);
    setIsUploading(true); // Show loading state

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/colorize/", {
        method: "POST",
        body: formData,
      });

      console.log("Response received:", response);

      if (response.ok) {
        const blob = await response.blob();
        setColorizedImage(URL.createObjectURL(blob));
        console.log("Colorized image received");
      } else {
        console.error("Upload failed:", response.statusText);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setIsUploading(false);
    }
  };

  // Calculate the main border container size
  const containerWidth = previewImage
    ? colorizedImage
      ? imageSize.width * 2 - 20 // 2 images side by side, but one shared border
      : imageSize.width
    : 500; // fallback if no preview

  const containerHeight = previewImage ? imageSize.height : 300; // fallback if no preview

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>

      {/* Row container: main images + star, plus the extra boxes on the right */}
      <div className="flex flex-row items-start">
        {/* Main border container + star button */}
        <div className="relative flex flex-row items-start">
          {/* Main border (label) - triggers file input */}
          <label
            htmlFor="file-upload"
            className="border-[10px] border-white rounded-xl flex"
            style={{
              width: `${containerWidth}px`,
              height: `${containerHeight}px`,
            }}
          >
            {/* If we have a preview, show it (and colorized) side by side */}
            {previewImage ? (
              <>
                {/* Preview (left side) */}
                <div
                  style={{
                    width: `${imageSize.width - 20}px`,
                    height: `${imageSize.height - 20}px`,
                  }}
                  className="flex items-center justify-center"
                >
                  <img
                    src={previewImage}
                    alt="Preview"
                    className="rounded-xl"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      objectFit: "cover",
                    }}
                  />
                </div>

                {/* Colorized image (right side), if available */}
                {colorizedImage && (
                  <div
                    style={{
                      width: `${imageSize.width - 20}px`,
                      height: `${imageSize.height - 20}px`,
                    }}
                    className="flex items-center justify-center"
                  >
                    <img
                      src={colorizedImage}
                      alt="Colorized"
                      className="rounded-xl"
                      style={{
                        maxWidth: "100%",
                        maxHeight: "100%",
                        objectFit: "cover",
                      }}
                    />
                  </div>
                )}
              </>
            ) : (
              // Otherwise, show the upload prompt
              <div className="w-full h-full flex flex-col items-center justify-center">
                <ArrowUpTrayIcon className="h-32 w-32 text-white" />
                <h2 className="text-5xl font-semibold mt-4">Upload an image</h2>
                <p className="text-xl mt-2">Click to browse</p>
                <p className="text-xl">or</p>
                <p className="text-xl">Drag and drop</p>
              </div>
            )}
          </label>

          {/* Hidden file input */}
          <input
            id="file-upload"
            type="file"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Star button (to colorize) - only if there's a preview and no colorized image yet */}
          {previewImage && !colorizedImage && (
            <button
              onClick={handleUpload}
              className="ml-4"
              disabled={isUploading}
              title="Click here to Colorize"
            >
              <img
                src="/images/Star.png"
                alt="Upload"
                className={`cursor-pointer w-50 h-50 transition-opacity ${
                  isUploading ? "opacity-50" : "opacity-100"
                }`}
              />
            </button>
          )}
        </div>

        {/* Right-side boxes: only show if we have a colorized image */}
        {colorizedImage && (
          <div className="flex flex-col items-center ml-4">
            {/* Box for the small colorized thumbnail */}
            <div className="border-[10px] border-white rounded-xl mb-4">
              <img
                src={colorizedImage}
                alt="Colorized Thumbnail"
                style={{ maxWidth: "128px", maxHeight: "128px" }}
                className="rounded"
              />
            </div>

            {/* Box for the download icon */}
            <div className="border-[5px] border-white rounded-xl flex flex-col items-center">
              {/* Download link for the full-size colorized image */}
              <a
                href={colorizedImage}
                download="colorized.png"
                title="Download colorized image"
              >
                <ArrowDownTrayIcon className="h-12 w-12 text-white" />
              </a>
            </div>

            {/* "Download" text below the icon's border */}
            <p className="mt-2 text-xl font-semibold">Download</p>
          </div>
        )}
      </div>

      <style>
        {`
          .text-gradient {
            background: linear-gradient(90deg, white, white, red, red, orange, yellow, green, blue, indigo, violet);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
        `}
      </style>
    </div>
  );
};

export default Home;
