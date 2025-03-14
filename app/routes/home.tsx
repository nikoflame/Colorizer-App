import React, { useState, useEffect } from "react";
import {
  ArrowUpTrayIcon,
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  HandThumbUpIcon as HandThumbUpIconOutline,
  HandThumbDownIcon as HandThumbDownIconOutline
} from "@heroicons/react/24/outline";
import { 
  HandThumbUpIcon as HandThumbUpIconSolid,
  HandThumbDownIcon as HandThumbDownIconSolid
 } from "@heroicons/react/24/solid";
import { set } from "mongoose";

const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 500, height: 300 });
  const [colorizedImage, setColorizedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [thumbsUp, setThumbsUp] = useState<boolean>(false);
  const [thumbsDown, setThumbsDown] = useState<boolean>(false);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [feedbackActive, setFeedbackActive] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<string>("");
  const [feedbackSuccess, setFeedbackSuccess] = useState<boolean>(false);
  const [feedbackError, setFeedbackError] = useState<boolean>(false);
  const [isFeedbackSubmitting, setIsFeedbackSubmitting] = useState<boolean>(false);
  const [isNoFeedbackSubmitting, setIsNoFeedbackSubmitting] = useState<boolean>(false);

  // Helper to process file uploads (from file input or drag and drop)
  const processFile = (file: File) => {
    setSelectedFile(file);
    setColorizedImage(null); // Reset colorized image if a new file is selected
    setThumbsUp(false);      // Reset thumbs up when a new file is uploaded

    const imageUrl = URL.createObjectURL(file);
    setPreviewImage(imageUrl);

    // Create an image element to get its original dimensions
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const { width, height } = img;
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

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.[0]) return;
    const file = event.target.files[0];
    processFile(file);
  };

  // Drag and drop event handlers
  const handleDragOver = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  };

  const handleDragEnter = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      const file = event.dataTransfer.files[0];
      processFile(file);
    }
  };

  // Handle Uploads
  const handleUpload = async () => {
    if (!selectedFile) return;

    console.log("Uploading file:", selectedFile.name);
    setIsUploading(true); // Show loading state

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("https://colorizer-app-1.onrender.com/colorize/", {
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

  // Handle thumbs up click
  const handleThumbUpClick = () => {
    setThumbsUp(!thumbsUp);
    if (!thumbsUp) {
      setThumbsDown(false);
      setFeedbackActive(false);
    }
  };

  // Handle thumbs down click
  const handleThumbDownClick = () => {
    setThumbsDown(!thumbsDown);
    if (!thumbsDown) {
      setThumbsUp(false);
      setFeedbackActive(!thumbsDown);
    }
  }

  const handleSubmitFeedback = () => {
    setIsFeedbackSubmitting(true);
    console.log('Submitting feedback:', feedback);
    fetch('https://colorizer-app-2.onrender.com/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedback }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          console.log('Feedback stored with ID:', data.id);
          setIsFeedbackSubmitting(false);
          setFeedbackSuccess(true);
        } else {
          console.error('Error storing feedback:', data.error);
          setIsFeedbackSubmitting(false);
          setFeedbackError(true);
        }
      })
      .catch((err) => console.error('Fetch error:', err));
      setIsFeedbackSubmitting(false);
  };
  
  const handleNoThankYou = () => {
    setIsNoFeedbackSubmitting(true);
    fetch('https://colorizer-app-2.onrender.com/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedback: 'N/A' }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          console.log('Feedback stored with ID:', data.id);
          setIsNoFeedbackSubmitting(false);
          setFeedback(''); // Clear the textarea after submission
          window.location.href = '/'; // Go back to the home page
        } else {
          console.error('Error storing feedback:', data.error);
          setIsNoFeedbackSubmitting(false);
          window.location.href = '/'; // Go back to the home page anyways
        }
      })
      .catch((err) => console.error('Fetch error:', err));
      setIsNoFeedbackSubmitting(false);
  };

  const handleReturnHomeFromFeedback = () => {
    setFeedback(''); // Clear the textarea after submission
    window.location.href = '/'; // Go back to the home page
  };

  // Calculate the main border container size
  const containerWidth = feedbackActive 
    ? 800 // fixed width for feedback section
      : previewImage
      ? colorizedImage
        ? imageSize.width * 2 - 20 // 2 images side by side, but one shared border
        : imageSize.width
      : 500; // fallback if no preview

  const containerHeight = feedbackActive ? 500 // fixed height for feedback section
    : previewImage ? imageSize.height : 300; // fallback if no preview

  // Display text for the feedback section
  const feedbackText = thumbsUp ? "Thanks!" : "How did we do?";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>

      {/* Row container: main images + star, plus the back button, plus the extra boxes on the right */}
      <div className="flex flex-row items-start">

        {/* Left-side boxes: only show if we have a preview */}
        {previewImage && (
          <div className="flex flex-col items-center mr-4">
            {/* Box for the back button icon */}
            <div className="border-[5px] border-white rounded-xl flex flex-col items-center">
              <a href="/" title="Back to Home">
                <ArrowLeftIcon className="h-12 w-12 text-white" />
              </a>
            </div>
            <p className="mt-2 text-xl font-semibold">Go Back</p>
          </div>
        )}

        {/* Main border container + star button */}
        <div className="relative flex flex-row items-start">
          {/* Main border (label) - triggers file input */}
          <label
            htmlFor="file-upload"
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-[10px] border-white rounded-xl flex ${dragActive ? 'bg-gray-400' : ''}`}
            style= {dragActive ? {
              width: `${containerWidth +10}px`,
              height: `${containerHeight +10}px`,
            } : {
              width: `${containerWidth}px`,
              height: `${containerHeight}px`,
            }}
          >
            {/* Feedback section (when thumbs down is solid) */}
            {
              feedbackActive ? (
                <div className="w-full h-full flex flex-col items-center justify-center">
                  <div className="border-[10px] border-white rounded-xl flex">
                    <div
                      className="flex items-center justify-center"
                    >
                      {/* Preview image with fallback */}
                      <img
                        src={previewImage ?? ""}
                        alt="Preview"
                        className="rounded-xl"
                        style={{
                          maxWidth: "128px",
                          maxHeight: "128px",
                          objectFit: "cover",
                        }}
                      />
                    </div>
                    <div
                      className="flex items-center justify-center"
                    >
                      {/* Colorized image with fallback */}
                      <img
                        src={colorizedImage ?? ""}
                        alt="Colorized"
                        className="rounded-xl"
                        style={{
                          maxWidth: "128px",
                          maxHeight: "128px",
                          objectFit: "cover",
                        }}
                      />
                    </div>
                  </div>
                  {/* Textbox section */}
                  <h2 className="text-2xl font-semibold mt-2">I'm sorry we didn't match your expectations!</h2>
                  <p className="text-2xl font-semibold mt-2">Please give us feedback so we can improve:</p>
                  <div className="mt-4 flex flex-col items-center space-y-4">
                    <textarea
                      className="w-full max-w-xl h-24 p-2 border border-gray-300 rounded"
                      placeholder="Your feedback here..."
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                    />

                    {/* Buttons row */}
                    <div className="flex space-x-4">
                      <button
                        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                        onClick={feedbackSuccess ? handleReturnHomeFromFeedback : handleSubmitFeedback}
                        disabled={isFeedbackSubmitting || isNoFeedbackSubmitting}
                      >
                        {feedbackSuccess 
                          ? "Thank you for your feedback! Click here to go back to the home page." 
                          : "SUBMIT"
                        }
                      </button>
                      { feedbackSuccess ? null : (
                          <button
                            className="bg-gray-400 text-white px-6 py-2 rounded hover:bg-gray-500"
                            onClick={handleNoThankYou}
                            disabled={isFeedbackSubmitting || isNoFeedbackSubmitting}
                          >
                            No, thank you <span className="text-sm ml-1">(return to home page)</span>
                          </button>
                        )
                      }
                    </div>

                    {/* Loading state */}
                    { isFeedbackSubmitting ? 
                      (
                        <div className="flex flex-col items-center">
                          <p className="text-2xl font-semibold mt-2">Please wait... Submitting feedback</p>
                          <img
                            src="public/images/loading.gif"
                            alt="Loading..."
                            className="cursor-pointer w-25 h-25"
                          />
                        </div>
                      ) : isNoFeedbackSubmitting ? (
                        <div className="flex flex-col items-center">
                          <p className="text-2xl font-semibold mt-2">Please wait... Sending thumbs-down to server</p>
                          <img
                            src="public/images/loading.gif"
                            alt="Loading..."
                            className="cursor-pointer w-25 h-25"
                          />
                        </div>
                      )
                    : null }
                  </div>
                </div>
              ) : previewImage ? (
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

          {/* Star button (to colorize) - with loading gif for processing */}
          {previewImage && !colorizedImage && (
            <button
              onClick={handleUpload}
              className="ml-4"
              disabled={isUploading}
              title="Click here to Colorize"
            >
              { isUploading ? (
                <img
                  src="public/images/loading.gif"
                  alt="Loading..."
                  className="cursor-pointer w-50 h-50"
                />
              ) : (
                <img
                  src="/images/Star.png"
                  alt="Upload"
                  className={`cursor-pointer w-50 h-50 transition-opacity`}
                />
              )}
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
              <a
                href={colorizedImage}
                download="colorized.png"
                title="Download colorized image"
              >
                <ArrowDownTrayIcon className="h-12 w-12 text-white" />
              </a>
            </div>

            <p className="mt-2 text-xl font-semibold">Download</p>

            {/* Feedback section */}
            <div className="flex flex-col items-center mt-4">
              <p className="mb-2 text-xl font-semibold">{feedbackText}</p>
              <div className="flex flex-row space-x-4">
                {thumbsUp && !thumbsDown ? (
                  <HandThumbUpIconSolid className="h-8 w-8 cursor-pointer" onClick={handleThumbUpClick} />
                ) : (
                  <HandThumbUpIconOutline className="h-8 w-8 cursor-pointer" onClick={handleThumbUpClick} />
                )}
                {thumbsDown && !thumbsUp ? (
                  <HandThumbDownIconSolid className="h-8 w-8 cursor-pointer" onClick={handleThumbDownClick} />
                ) : (
                  <HandThumbDownIconOutline className="h-8 w-8 cursor-pointer" onClick={handleThumbDownClick} />
                )}
              </div>
            </div>
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
