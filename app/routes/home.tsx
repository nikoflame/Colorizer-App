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

      // Increase border size by 10px in each direction (20px total size increase)
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
        setColorizedImage(URL.createObjectURL(blob)); // Replace preview with colorized image
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

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>

      {/* Upload Box (Border Stays, Content Changes, Dynamic Size) */}
      <div className="relative flex items-center">
        <label 
          htmlFor="file-upload"
          className={`cursor-pointer rounded-xl text-center flex items-center justify-center ${
            previewImage ? "border-[10px]" : "border-10"
          } border-white`}
          style={{ width: `${imageSize.width}px`, height: `${imageSize.height}px` }} // Dynamically update size
        >
          {colorizedImage ? (
            // Show Colorized Image
            <img src={colorizedImage} alt="Colorized" className="w-full h-full object-cover rounded-xl" />
          ) : previewImage ? (
            // Show Preview Before Upload (Resized)
            <img src={previewImage} alt="Preview" className="rounded-xl" style={{ maxWidth: "512px", maxHeight: "512px", width: `${imageSize.width - 20}px`, height: `${imageSize.height - 20}px` }} />
          ) : (
            // Default Upload Prompt
            <div className="flex flex-col items-center">
              <ArrowUpTrayIcon className="h-32 w-32 text-white" />
              <h2 className="text-5xl font-semibold mt-4">Upload an image</h2>
              <p className="text-xl mt-2">Click to browse</p>
              <p className="text-xl">or</p>
              <p className="text-xl">Drag and drop</p>
            </div>
          )}
        </label>

        {/* Hidden File Input (Triggers on Click) */}
        <input
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Upload Button (Appears Only When an Image is Selected) */}
        {previewImage && !colorizedImage && (
          <button 
            onClick={handleUpload}
            className="ml-4"
            disabled={isUploading}
          >
            <img 
              src="/images/Star.png"
              alt="Upload"
              className={`cursor-pointer w-50 h-50 transition-opacity ${isUploading ? 'opacity-50' : 'opacity-100'}`}
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
