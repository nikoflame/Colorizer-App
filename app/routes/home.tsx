import React, { useState, useEffect } from "react";
import { ArrowUpTrayIcon } from "@heroicons/react/24/outline";

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

      // We add 20 to each dimension to account for a 10px border all around
      setImageSize({ width: Math.round(newWidth) + 20, height: Math.round(newHeight) + 20 });
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

  // Determine the container width:
  // - If we have both preview and colorized images, make room for both side by side.
  // - Otherwise, just room for the preview or (if no preview) a default size.
  const containerWidth = previewImage
    ? colorizedImage
      // For two images side by side, we do: imageSize.width * 2 - 20
      // Explanation:
      //   - imageSize.width includes 20px for border (10px each side).
      //   - For two images in a single border, we only add one extra "image width minus its border".
      //   - So total is (width + (width - 20)) = 2*width - 20.
      ? imageSize.width * 2 - 20
      : imageSize.width
    : 500; // fallback if no preview

  const containerHeight = previewImage ? imageSize.height : 300; // fallback if no preview

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>

      {/* Wrap everything (border + star) in a flex row so the star can be on the right */}
      <div className="relative flex items-center">
        {/* Label for file input so user can click anywhere in the border to upload */}
        <label
          htmlFor="file-upload"
          className="border-[10px] border-white rounded-xl flex"
          style={{
            width: `${containerWidth}px`,
            height: `${containerHeight}px`,
          }}
        >
          {/* If there's a preview, show either preview alone or preview + colorized side by side */}
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

              {/* Colorized (right side), only if available */}
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
            // No preview yet: show upload prompt
            <div className="w-full h-full flex flex-col items-center justify-center">
              <ArrowUpTrayIcon className="h-32 w-32 text-white" />
              <h2 className="text-5xl font-semibold mt-4">Upload an image</h2>
              <p className="text-xl mt-2">Click to browse</p>
              <p className="text-xl">or</p>
              <p className="text-xl">Drag and drop</p>
            </div>
          )}
        </label>

        {/* Hidden file input (triggered by label) */}
        <input
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Star button on the right, only shown if there's a preview but no colorized image yet */}
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