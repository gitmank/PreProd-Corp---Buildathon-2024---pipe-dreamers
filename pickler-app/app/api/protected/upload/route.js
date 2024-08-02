import { NextResponse } from "next/server";
import { bucket } from "@/lib/connectToGCS";
import connectToDB from "@/lib/connectToDB";
import File from "@/models/FileModel";
import { cookies } from "next/headers";
import jwt from "jsonwebtoken";

const COOKIE_NAME = 'user-auth';
const URL_EXPIRY = 5 * 60 * 1000; // 5 minutes

export async function POST(req, res) {
    try {
        await connectToDB();
        const { name, type, size } = await req.json();
        const token = cookies(req).get('user-auth').value;
        const { email } = jwt.verify(token, process.env.JWT_SECRET);
        const objectName = `${name}-${Math.random().toString(16).substring(6)}`;
        // get presigned url
        const [url] = await bucket.file(objectName).getSignedUrl({
            version: "v4",
            action: "write",
            expires: new Date().getTime() + URL_EXPIRY,
            contentType: 'application/octet-stream',
            extensionHeaders: {
                "x-upload-content-length": size,
            },
        });
        const file = new File({ name, objectName, owner: email, size, type });
        await file.save();
        return NextResponse.json({ url }, { status: 200 });
    } catch (error) {
        console.error('upload error', error);
        return NextResponse.json({ error: 'internal error' }, { status: 500 });
    }
}