"use client";

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

export default function Home() {
  const handleSubmit = () => {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    console.log(email, password);
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
