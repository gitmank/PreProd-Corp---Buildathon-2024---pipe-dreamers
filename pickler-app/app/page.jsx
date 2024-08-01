"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const STATUS = {
  LOADING: "loading",
  ERROR: "error",
  SUCCESS: "success",
  DEFAULT: "",
};

export default function Home() {
  const [status, setStatus] = useState(STATUS.DEFAULT);
  const router = useRouter();

  // event handlers
  const handleSubmit = async () => {
    try {
      setStatus(STATUS.LOADING);
      const emailField = document.getElementById("email");
      const passwordField = document.getElementById("password");
      // validate email and password
      const result = validateFields(emailField.value, passwordField.value);
      if (!result) {
        setStatus(STATUS.ERROR);
        return;
      }
      const response = await fetch("/api/public/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: emailField.value,
          password: passwordField.value,
        }),
      });
      if (response.ok) {
        setStatus(STATUS.SUCCESS);
        router.push("/dashboard");
      } else {
        setStatus(STATUS.ERROR);
      }
    } catch (error) {
      console.error("Error submitting form", error);
      setStatus(STATUS.ERROR);
    }
  };

  // helper functions
  const validateFields = (email, password) => {
    if (!email || !password) {
      alert("Please enter a valid email and password");
      return false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const strongPasswordRegex = /\w{8,}/;
    if (!emailRegex.test(email)) {
      alert("Please enter a valid email");
      return false;
    }
    if (!strongPasswordRegex.test(password)) {
      alert("Please enter a password that is at least 8 characters long");
      return false;
    }
    return true;
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-start gap-8 p-24">
      <h1 className="text-4xl font-bold">Welcome to Pickler!</h1>
      <Image src="/picklerick.png" width={100} height={100} />
      <Card className="mx-auto max-w-sm">
        <CardHeader>
          <CardTitle className="text-xl">Authenticate</CardTitle>
          <CardDescription>
            Enter your information to login/signup
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {/* <div className="grid gap-2">
              <Label htmlFor="first-name">Name</Label>
              <Input id="first-name" placeholder="Max" required />
            </div> */}
            <div className="grid gap-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="m@example.com"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" />
            </div>
            <Button onClick={handleSubmit} className="w-full mt-4">
              Submit
            </Button>
            <p className="self-center text-center">{status}</p>
            {/* <Button variant="outline" className="w-full">
              Sign up with GitHub
            </Button> */}
          </div>
          {/* <div className="mt-4 text-center text-sm">
            Already have an account?{" "}
            <Link href="#" className="underline">
              Sign in
            </Link>
          </div> */}
        </CardContent>
      </Card>
    </main>
  );
}
