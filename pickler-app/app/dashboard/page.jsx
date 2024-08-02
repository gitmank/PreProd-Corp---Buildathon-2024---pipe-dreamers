"use client";
import { useAuth } from "@/hooks/useAuth";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import Image from "next/image";
import { File, ListFilter, MoreHorizontal, PlusCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Dashboard() {
  const [user, loading] = useAuth();
  const [files, setFiles] = useState([]);
  const router = useRouter();

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await fetch("/api/protected/files");
        if (response.ok) {
          const data = await response.json();
          setFiles(data.files);
        }
      } catch (error) {
        console.error("Error fetching files", error);
      }
    };

    fetchFiles();
  }, []);

  return (
    <div className="flex min-h-screen w-full flex-col bg-muted/40">
      <div className="flex flex-col sm:gap-4 sm:py-4">
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-4 border-b bg-background px-4 sm:static sm:h-auto sm:border-0 sm:bg-transparent sm:px-6">
          <h1>Pickler 🥒</h1>
          <h1>Hi {user?.email.split("@")[0]} 👋</h1>
        </header>
        <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0 md:gap-8">
          <Tabs defaultValue="files">
            <div className="flex items-center">
              <TabsList>
                <TabsTrigger value="files">Files</TabsTrigger>
                <TabsTrigger value="models">Models</TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="files">
              <div className="ml-auto flex items-center gap-2 my-2">
                <Button
                  onClick={() => router.push("/upload")}
                  size="sm"
                  className="h-8 gap-1"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">
                    Upload File
                  </span>
                </Button>
              </div>
              <Card x-chunk="dashboard-06-chunk-0">
                <CardHeader>
                  <CardTitle>Files</CardTitle>
                  <CardDescription>upload and manage files</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="hidden w-[100px] sm:table-cell">
                          <span className="sr-only">Image</span>
                        </TableHead>
                        <TableHead>Name</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead className="hidden md:table-cell">
                          Size
                        </TableHead>
                        <TableHead className="hidden md:table-cell">
                          Type
                        </TableHead>
                        <TableHead className="hidden md:table-cell">
                          Actions
                        </TableHead>
                        <TableHead>
                          <span className="sr-only">Actions</span>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {files &&
                        files.map((file) => (
                          <FileRow
                            key={file._id}
                            file={file}
                            setFiles={setFiles}
                          />
                        ))}
                    </TableBody>
                  </Table>
                </CardContent>
                <CardFooter>
                  <div className="text-xs text-muted-foreground">
                    Your uploaded files will appear here.
                  </div>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  );
}

const FileRow = ({ file, setFiles }) => {
  const handleDelete = async (id) => {
    try {
      await fetch(`/api/protected/files?id=${id}`, {
        method: "DELETE",
      });
      setFiles((prevFiles) => prevFiles.filter((file) => file._id !== id));
    } catch (error) {
      console.error("Error deleting file", error);
    }
  };
  return (
    <TableRow>
      <TableCell className="hidden sm:table-cell">
        <Image
          alt="file icon"
          className="aspect-square rounded-md object-cover"
          height="64"
          src="/placeholder.png"
          width="64"
        />
      </TableCell>
      <TableCell className="font-medium">{file.name}</TableCell>
      <TableCell>
        <Badge variant="">{file.status}</Badge>
      </TableCell>
      <TableCell className="hidden md:table-cell">
        {file.size > 1000000
          ? `${parseFloat(file.size / 1000000).toFixed(2)} MB`
          : `${parseFloat(file.size / 1000000).toFixed(2)} KB`}
      </TableCell>
      <TableCell className="hidden md:table-cell">
        {file.name.split(".").pop()}
      </TableCell>
      <TableCell>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button aria-haspopup="true" size="icon" variant="ghost">
              <MoreHorizontal className="h-4 w-4" />
              <span className="sr-only">Toggle menu</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem>Clean</DropdownMenuItem>
            <DropdownMenuItem>Train</DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => handleDelete(file._id)}
              className="text-red-400"
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
};
