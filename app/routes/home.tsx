import React from "react";
import { ArrowUpTrayIcon } from "@heroicons/react/24/outline";

const Home: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0E005F] text-white">
      <h1 className="text-9xl mb-6">
        AI Col<span className="text-gradient">orizer</span>
      </h1>
      <div className="border-10 border-white rounded-xl p-5 w-200 text-center items-center justify-center flex flex-col">
        <ArrowUpTrayIcon className="h-32 w-32 text-white" />
        <h2 className="text-5xl font-semibold mt-4">Upload an image</h2>
        <div className="h-16"></div>
        <p className="text-xl mt-2">Click to browse</p>
        <p className="text-xl">or</p>
        <p className="text-xl">Drag and drop</p>
        <div className="h-8"></div>
      </div>
      <style>
        {`
          .text-gradient {
            background: linear-gradient(90deg, white, red, orange, yellow, green, blue, indigo, violet);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
        `}
      </style>
    </div>
  );
};

export default Home;
