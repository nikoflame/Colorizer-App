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
  const [isImageTooLarge, setIsImageTooLarge] = useState<boolean>(false);
  const [secondaryFeedbackActive, setSecondaryFeedbackActive] = useState<boolean>(false);
  const [downloadFileName, setDownloadFileName] = useState<string>("");
  const [thumbsUpFeedbackActive, setThumbsUpFeedbackActive] = useState<boolean>(false);
  const [isMobile, setIsMobile] = useState<boolean>(false);

  // Detect mobile screen sizes
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // Helper to process file uploads (from file input or drag and drop)
  const processFile = (file: File) => {
    setSelectedFile(file);
    setColorizedImage(null); // Reset colorized image if a new file is selected
    setThumbsUp(false);      // Reset thumbs up when a new file is uploaded
  
    // Set the default download file name using the file's name (without extension)
    const baseName = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
    setDownloadFileName(`${baseName}_colorized`);
  
    const imageUrl = URL.createObjectURL(file);
    setPreviewImage(imageUrl);
  
    // Create an image element to get its original dimensions
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const { width, height } = img;
      const maxDimension = 512;
  
      if (width > maxDimension || height > maxDimension) {
        // Mark the image as too large and prompt the user for resizing
        setIsImageTooLarge(true);
      } else {
        // If within limits, calculate size for display and proceed
        let newWidth = width;
        let newHeight = height;
        if (width > height) {
          newWidth = maxDimension;
          newHeight = (height / width) * maxDimension;
        } else {
          newHeight = maxDimension;
          newWidth = (width / height) * maxDimension;
        }
        // Add 20 to each dimension for a 10px border
        setImageSize({
          width: Math.round(newWidth) + 20,
          height: Math.round(newHeight) + 20,
        });
        setIsImageTooLarge(false);
      }
    };
  };  

  const resize = async () => {
    if (!selectedFile || !previewImage) return;
  
    const img = new Image();
    img.src = previewImage;
    await new Promise((resolve) => {
      img.onload = resolve;
    });
    
    const maxDimension = 512;
    let newWidth = img.width;
    let newHeight = img.height;
    if (img.width > maxDimension || img.height > maxDimension) {
      if (img.width > img.height) {
        newWidth = maxDimension;
        newHeight = (img.height / img.width) * maxDimension;
      } else {
        newHeight = maxDimension;
        newWidth = (img.width / img.height) * maxDimension;
      }
    }
    
    const canvas = document.createElement("canvas");
    canvas.width = newWidth;
    canvas.height = newHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("Could not create canvas context");
      return;
    }
    ctx.drawImage(img, 0, 0, newWidth, newHeight);
    
    // Convert the canvas content to a Blob
    canvas.toBlob(async (blob) => {
      if (blob) {
        // Create a new File object from the blob so it can be used in FormData
        const resizedFile = new File([blob], selectedFile.name, { type: blob.type });
        // Update state with the resized file and preview image
        setSelectedFile(resizedFile);
        const resizedUrl = URL.createObjectURL(blob);
        setPreviewImage(resizedUrl);
        // Update the display size state (adding the border)
        setImageSize({
          width: Math.round(newWidth) + 20,
          height: Math.round(newHeight) + 20,
        });
        // Clear the "image too large" flag and upload
        setIsImageTooLarge(false);
      }
    }, selectedFile.type);
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
      setThumbsUpFeedbackActive(true);
    } else {
      setThumbsUpFeedbackActive(false)
    }
  };

  // Handle thumbs down click
  const handleThumbDownClick = () => {
    setThumbsDown(!thumbsDown);
    if (!thumbsDown) {
      setThumbsUp(false);
      setFeedbackActive(true);
    } else {
      setFeedbackActive(false)
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
          setFeedbackSuccess(true);
        } else {
          console.error('Error storing feedback:', data.error);
          setFeedbackError(true);
        }
      })
      .catch((err) => console.error('Fetch error:', err))
      .finally(() => {
        setIsFeedbackSubmitting(false);
      });
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
          setFeedback(''); // Clear the textarea after submission
          window.location.href = '/'; // Go back to the home page
        } else {
          console.error('Error storing feedback:', data.error);
          window.location.href = '/'; // Go back to the home page anyways
        }
      })
      .catch((err) => console.error('Fetch error:', err))
      .finally(() => {
        setIsNoFeedbackSubmitting(false);
      });
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

  // Determine the style for the label container – use inline sizes only on desktop
  const labelStyle = !isMobile
    ? dragActive
      ? { width: `${containerWidth + 10}px`, height: `${containerHeight + 10}px` }
      : { width: `${containerWidth}px`, height: `${containerHeight}px` }
    : {};

  // Display text for the feedback section
  const feedbackText = thumbsUp ? "Thanks!" : "How did we do?";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-5xl md:text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>

      {/* Main container: stack vertically on mobile, row on desktop */}
      <div className="flex flex-col md:flex-row items-start">

        {/* Left-side boxes: only show if we have a preview or secondary feedback */}
        {(previewImage || secondaryFeedbackActive) && (
          <div className="flex flex-col items-center mb-4 md:mb-0 md:mr-4">
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
        <div className="relative flex flex-col md:flex-row items-start">
          {/* Main border (label) - triggers file input */}
          <label
            htmlFor={ (secondaryFeedbackActive || feedbackActive) ? "" : "file-upload"}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-[10px] border-white rounded-xl flex ${dragActive ? 'bg-gray-400' : ''} ${isMobile ? 'w-11/12' : ''}`}
            style={labelStyle}
          >
            {/* Feedback section */}
            {
              (secondaryFeedbackActive || feedbackActive) ? (
                <div className="w-full h-full flex flex-col items-center justify-center">
                  {feedbackActive && (
                    // Image preview and colorized image
                    <div className={`border-[10px] border-white rounded-xl flex ${isMobile ? 'flex-col items-center gap-4' : ''}`}>
                      <div className="flex items-center justify-center">
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
                      <div className="flex items-center justify-center">
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
                  )}
                  {/* Textbox section */}
                  {feedbackActive ? (
                    <div>
                      <h2 className="text-2xl font-semibold mt-2">I'm sorry we didn't match your expectations!</h2>
                      <p className="text-2xl font-semibold mt-2">Please give us feedback so we can improve:</p>
                    </div>
                  ) : (
                    <h2 className="text-2xl font-semibold mt-2">We'd love to hear your feedback!</h2>
                  )}
                  <div className="mt-4 flex flex-col items-center space-y-4">
                    <textarea
                      className="w-full max-w-md md:max-w-xl h-24 p-2 border border-gray-300 rounded"
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
                            onClick={feedbackActive ? handleNoThankYou : handleReturnHomeFromFeedback}
                            disabled={isFeedbackSubmitting || isNoFeedbackSubmitting}
                          >
                            No, thank you <span className="text-sm ml-1">(return to home page)</span>
                          </button>
                        )
                      }
                    </div>

                    {/* Loading state */}
                    {(isFeedbackSubmitting || isNoFeedbackSubmitting) && (
                      <div className="flex flex-col items-center">
                        <p className="text-2xl font-semibold mt-2">
                          {isFeedbackSubmitting
                            ? "Please wait... Submitting feedback"
                            : "Please wait... Sending thumbs-down as feedback"}
                        </p>
                        <img
                          src="/images/loading.gif"
                          alt="Loading..."
                          className="cursor-pointer w-24 h-24"
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : previewImage ? (
                <>
                  {/* Preview (left side) */}
                  <div
                    style={
                      !isMobile ? { 
                        width: `${imageSize.width - 20}px`,
                        height: `${imageSize.height - 20}px`,
                      } : {}
                    }
                    className={`flex items-center justify-center ${isMobile ? 'w-full' : ''}`}
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
                      style={
                        !isMobile? {
                          width: `${imageSize.width - 20}px`,
                          height: `${imageSize.height - 20}px`,
                        }: {}
                      }
                      className={`flex items-center justify-center ${isMobile ? 'w-full' : ''}`}
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
          {isImageTooLarge ? (
            <div className="mt-4 p-4 bg-red-500 text-white rounded">
              <p>Your image is too large!</p>
              <p>Our free service does not allow images beyond a size of 512x512 pixels.</p>
              <p> </p>
              <p>Would you like us to resize it for you?</p>
              <div className="flex flex-col space-x-4">
                <button
                  onClick={resize}
                  className="mt-2 bg-blue-600 px-4 py-2 rounded hover:bg-blue-700"
                >
                  Resize my image for me
                </button>
                <button
                  className="bg-gray-400 text-white px-6 py-2 rounded hover:bg-gray-500"
                  onClick={handleReturnHomeFromFeedback}
                >
                  No, thank you <span className="text-sm ml-1">(return to home page)</span>
                </button>
              </div>
            </div>
          ) : previewImage && !colorizedImage && (
              <button
                onClick={handleUpload}
                className="mt-4 md:ml-4"
                disabled={isUploading}
                title="Click here to Colorize"
              >
                { isUploading ? (
                  <div className="flex flex-col items-center">
                    <img
                      src="/images/loading.gif"
                      alt="Loading..."
                      className="cursor-pointer w-50 h-50"
                    />
                    <p className="text-xl font-semibold mt-2">Colorizing, please wait...</p>
                    <p className="text-xl mt-2">This may take up to three minutes</p>
                    <p className="text-xl mt-2">if the server has been idle for a while...</p>
                  </div>
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
          <div className="flex flex-col items-center mt-4 md:mt-0 md:ml-4">

            <p className="text-sm md:ml-1">Change file name here:</p>

            {/* File name input above the thumbnail */}
            <div className="flex flex-row items-center md:ml-4">
              <div className="mb-2">
                <input
                  type="text"
                  value={downloadFileName}
                  onChange={(e) => setDownloadFileName(e.target.value)}
                  className="px-2 py-1 rounded"
                />
              </div>
              <p className="ml-2 text-xl font-semibold">.png</p>
            </div>

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
                download={downloadFileName + ".png"}
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

            {/* ThumbsUp Feedback section, hidden until thumbs up */}
            {thumbsUpFeedbackActive && (
              <div className="flex flex-col items-center mt-4">
                <p className="mb-2 text-xl">Would you like to leave a comment?</p>
                <div className="flex flex-col md:flex-row items-center space-y-4">
                  <textarea
                    className="w-full max-w-md md:max-w-xl h-24 p-2 border border-gray-300 rounded"
                    placeholder="Your feedback here..."
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                  />
                  <button
                    className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                    onClick={handleSubmitFeedback}
                    disabled={isFeedbackSubmitting || isNoFeedbackSubmitting || feedbackSuccess}
                  >{isFeedbackSubmitting 
                    ? "Submitting, please wait..." 
                    : feedbackSuccess
                    ? "Thanks!"
                    : "Submit"
                  }</button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Feedback button */}
      {!colorizedImage && (
        <button 
          onClick={() => setSecondaryFeedbackActive(true)}
          className="mt-4 p-4 bg-white text-black rounded"
          >
            Click here to leave feedback for a previous experience, or anything else!
        </button>
      )}

      {/* Loading time warning */}
      <div className="mt-4 p-2 bg-red-500 text-white rounded text-center">
        <p className="mt-4 text-3xl">This is a free service!</p>
        <p>----------</p>
        <p className="mt-2 text-xl">Be aware that loading times may take up to 3 minutes</p>
        <p className="mt-2 text-xl">for the first request after the server has been idle for a while</p>
        <p>----------</p>
        <p className="mt-2 text-xl">Thank you for your understanding!</p>
      </div>
  
      {/* Footer */}
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
