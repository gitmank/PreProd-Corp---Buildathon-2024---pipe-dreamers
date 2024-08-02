"use client";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useEffect, useState } from "react";

const STATUS = {
  DEFAULT: "",
  UPLOADING: "uploading ⏳",
  UPLOADED: "uploaded 🟢",
  ERROR: "error 🔴",
};

export default function LoginForm() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [status, setStatus] = useState(STATUS.DEFAULT);
  const [uploadURL, setUploadURL] = useState(null);
  const [fileID, setFileID] = useState(null);

  const triggerInput = () => {
    const input = document.getElementById("file-input");
    input.click();
  };

  useEffect(() => {
    if (!uploadedFile) return;
    setStatus(STATUS.UPLOADING);
    const getUploadURL = async () => {
      try {
        const response = await fetch("/api/protected/upload", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: uploadedFile.name,
            size: uploadedFile.size,
            type: uploadedFile.type,
          }),
        });
        if (response.ok) {
          const data = await response.json();
          setUploadURL(data.url);
          setFileID(data.id);
        }
      } catch (error) {
        console.error("Error fetching upload URL", error);
        setStatus(STATUS.ERROR);
        alert("Error uploading file");
      }
    };
    getUploadURL();
  }, [uploadedFile]);

  useEffect(() => {
    if (!uploadURL) return;
    const uploadFile = async () => {
      try {
        const response = await fetch(uploadURL, {
          method: "PUT",
          body: uploadedFile,
          headers: {
            "Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": uploadedFile.size,
          },
        });
        if (response.ok) {
          setStatus(STATUS.UPLOADED);
          queueFile();
          alert("File uploaded successfully");
        }
      } catch (error) {
        console.error("Error uploading file", error);
        alert("Error uploading file");
        setStatus(STATUS.ERROR);
      }
    };
    uploadFile();
  }, [uploadURL]);

  const queueFile = async () => {
    try {
      await fetch(`/api/protected/files/extract?id=${fileID}`);
    } catch (error) {
      console.error("Error queuing file", error);
    }
  };

  return (
    <main className="flex items-center justify-center min-h-screen w-screen bg-[url('/picklerick.png')] bg-center">
      <Card className="w-full max-w-sm border-gray-400">
        <CardHeader>
          <CardTitle className="text-2xl">Upload</CardTitle>
          <CardDescription>accepts .xlsx | .xls | .csv</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {uploadedFile ? (
            <FileCard
              file={uploadedFile}
              handleClear={() => setUploadedFile(null)}
              status={status}
            />
          ) : (
            <UploadCard
              triggerInput={triggerInput}
              setUploadedFile={setUploadedFile}
            />
          )}
        </CardContent>
        <CardFooter className="mt-4">
          <a
            href="/dashboard"
            className="w-max bg-white p-1 px-2 text-black rounded-md"
          >
            Exit
          </a>
        </CardFooter>
      </Card>
    </main>
  );
}

// page sepcific components
const FileCard = ({ file, status, handleClear }) => {
  return (
    <div className="flex flex-col gap-4 justify-center items-center w-full h-max p-1">
      <p className="text-center text-base">
        {file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}
      </p>
      <div className="flex flex-col relative gap-2 text-center justify-center items-center w-max h-max border-white border-2 p-3 rounded-md">
        <button
          onClick={handleClear}
          className="flex w-6 h-6 justify-center items-center rounded-full pb-1 bg-white text-red-400 absolute -top-3 -right-3 hover:bg-red-400 hover:text-white duration-100"
        >
          x
        </button>
        <p className="text-4xl">{"📊"}</p>
      </div>
      <p>
        {file.size > 1000000
          ? `${(file.size / 1000000).toFixed(2)} MB`
          : `${(file.size / 1000).toFixed(2)} KB`}
      </p>
      <p>{status}</p>
    </div>
  );
};

const UploadCard = ({ triggerInput, setUploadedFile }) => {
  return (
    <div className="flex flex-col gap-4 items-center justify-center pt-4">
      <p
        onClick={triggerInput}
        className="flex items-center justify-center bg-white text-black rounded-full w-20 h-20 text-2xl hover:scale-105 active:scale-95 duration-100 select-none"
      >
        +
      </p>
      <input
        type="file"
        name="file-input"
        id="file-input"
        accept=".xlsx,.xls,.csv"
        max={1}
        className="hidden"
        onChange={(e) => setUploadedFile(e.target.files[0])}
      />
    </div>
  );
};
