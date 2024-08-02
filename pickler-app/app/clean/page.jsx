"use client";
import { useState, useEffect, Suspense } from "react";
import { useAuth } from "@/hooks/useAuth";
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
      const response = await fetch("/api/protected/files/clean", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          id,
          ...form,
        }),
      });
      if (response.ok) {
        alert("Data queued for cleaning");
      }
    } catch (error) {
      console.error("Error cleaning data", error);
    }
  };

  return (
    <Card className="w-full max-w-sm border-gray-400">
      <CardHeader>
        <CardTitle className="text-2xl">Clean</CardTitle>
        <CardDescription>use options to clean your data</CardDescription>
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
                onClick={() => setForm({ ...form, target: feature })}
                className={
                  form.target == feature
                    ? "bg-green-400 duration-100 hover:bg-green-400"
                    : "bg-white duration-100 hover:bg-white"
                }
              >
                {feature}
              </Badge>
            ))}
        </div>
        <div className="flex flex-col gap-2">
          <label htmlFor="encoding">Encoding</label>
          <select
            name="encoding"
            id="encoding"
            value={form.encoding}
            className="bg-gray-700"
            onChange={(e) => setForm({ ...form, encoding: e.target.value })}
          >
            <option value="onehot">One Hot Encoding</option>
            <option value="label">Label Encoding</option>
          </select>
        </div>
        {/* remove features */}
        <div className="flex flex-col gap-2">
          <p>Remove Features</p>
          {
            <div className="flex flex-row flex-wrap gap-1">
              {fileData?.features &&
                fileData?.features.map((feature) => (
                  <Badge
                    key={feature}
                    onClick={() =>
                      setForm({
                        ...form,
                        remove: form.remove.includes(feature)
                          ? form.remove.filter((item) => item !== feature)
                          : [...form.remove, feature],
                      })
                    }
                    className={
                      form.remove.includes(feature)
                        ? "bg-red-400 duration-100 hover:bg-red-400"
                        : "bg-white duration-100 hover:bg-white"
                    }
                  >
                    {feature}
                  </Badge>
                ))}
            </div>
          }
        </div>
        <p>Numerical Features</p>
        <div className="flex flex-row flex-wrap gap-1">
          {fileData?.features &&
            fileData?.features.map((feature) => (
              <Badge
                key={feature}
                onClick={() =>
                  setForm({
                    ...form,
                    numerical: form.numerical.includes(feature)
                      ? form.numerical.filter((item) => item !== feature)
                      : [...form.numerical, feature],
                  })
                }
                className={
                  form.numerical.includes(feature)
                    ? "bg-blue-400 duration-100 hover:bg-blue-400"
                    : "bg-white duration-100 hover:bg-white"
                }
              >
                {feature}
              </Badge>
            ))}
        </div>
        <p>Categorical Features</p>
        <div className="flex flex-row flex-wrap gap-1">
          {fileData?.features &&
            fileData?.features.map((feature) => (
              <Badge
                key={feature}
                onClick={() =>
                  setForm({
                    ...form,
                    categorical: form.categorical.includes(feature)
                      ? form.categorical.filter((item) => item !== feature)
                      : [...form.categorical, feature],
                  })
                }
                className={
                  form.categorical.includes(feature)
                    ? "bg-blue-400 duration-100 hover:bg-blue-400"
                    : "bg-white duration-100 hover:bg-white"
                }
              >
                {feature}
              </Badge>
            ))}
        </div>
        <Button onClick={handleSubmit}>Submit</Button>
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
