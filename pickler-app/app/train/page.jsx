"use client";
import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";

export default function CleanPage() {
  return (
    <main className="flex items-center justify-center min-h-screen w-screen bg-[url('/picklerick.png')] bg-center">
      <Suspense fallback={<div>Loading...</div>}>
        <CleanForm />
      </Suspense>
    </main>
  );
}

const CleanForm = () => {
  const searchParams = useSearchParams();
  const id = searchParams.get("id");
  const [fileData, setFileData] = useState(null);
  const [target, setTarget] = useState(null);
  const [form, setForm] = useState({
    encoding: "onehot",
    remove: [],
    target: "id",
    numerical: [],
    categorical: [],
  });

  useEffect(() => {
    const getFileData = async () => {
      try {
        const response = await fetch(`/api/protected/files?id=${id}`);
        if (response.ok) {
          const data = await response.json();
          setFileData(data.fileData);
        }
      } catch (error) {
        console.error("Error fetching file", error);
      }
    };
    if (!id) return;
    getFileData();
  }, []);

  const handleSubmit = async () => {
    try {
      const response = await fetch("/api/protected/files/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          id,
          form: { ...form },
        }),
      });
      if (response.ok) {
        alert("Data queued for cleaning");
      }
    } catch (error) {
      console.error("Error cleaning data", error);
    }
  };

  const KERNEL_OPTIONS = ["linear", "poly", "rbf", "sigmoid"];
  const GAMMA_OPTIONS = ["auto", "scale"];

  return (
    <Card className="w-full max-w-sm border-gray-400">
      <CardHeader>
        <CardTitle className="text-2xl">Train</CardTitle>
        <CardDescription>use a cleaned file to train your data</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div>
          {fileData ? (
            <FileCard file={fileData} handleClear={() => setFileData(null)} />
          ) : (
            <p>No file found</p>
          )}
        </div>
        <p className="text-lg self-start">Target Feature</p>
        <div className="flex flex-row flex-wrap gap-1">
          {fileData?.features &&
            fileData?.features.map((feature) => (
              <Badge
                key={feature}
                onClick={() => setTarget(feature)}
                className={
                  target == feature
                    ? "bg-green-400 duration-100 hover:bg-green-400"
                    : "bg-white duration-100 hover:bg-white"
                }
              >
                {feature}
              </Badge>
            ))}
        </div>
        <Accordion type="single" collapsible>
          <AccordionItem value="item-1">
            <AccordionTrigger>SVM</AccordionTrigger>
            <AccordionContent className="flex flex-col gap-2">
              <div className="flex flex-row gap-1">
                <p>Kernel&nbsp;</p>
                {
                  // kernel pick out of 4
                  KERNEL_OPTIONS.map((kernel) => (
                    <Badge
                      key={kernel}
                      onClick={() => setForm({ ...form, kernel })}
                      className={
                        form.kernel == kernel
                          ? "bg-green-400 duration-100 hover:bg-green-400"
                          : "bg-white duration-100 hover:bg-white"
                      }
                    >
                      {kernel}
                    </Badge>
                  ))
                }
              </div>
              <div className="flex flex-row gap-1">
                <p>Gamma</p>
                {
                  // gamma pick out of 2
                  GAMMA_OPTIONS.map((gamma) => (
                    <Badge
                      key={gamma}
                      onClick={() => setForm({ ...form, gamma })}
                      className={
                        form.gamma == gamma
                          ? "bg-green-400 duration-100 hover:bg-green-400"
                          : "bg-white duration-100 hover:bg-white"
                      }
                    >
                      {gamma}
                    </Badge>
                  ))
                }
              </div>
              <Input
                type="text"
                name="svm-c"
                id="svm-c"
                placeholder="C-value"
                className="w-full"
                onChange={(e) => setForm({ ...form, c: e.target.value })}
              />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
        <Button onClick={handleSubmit}>Train</Button>
      </CardContent>
    </Card>
  );
};

const FileCard = ({ file, handleClear }) => {
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
      <p>
        {file?.rows || "undef"} rows x {file?.columns || "undef"} cols
      </p>
    </div>
  );
};
