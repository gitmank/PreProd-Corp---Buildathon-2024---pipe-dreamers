import { NextResponse } from "next/server";
import connectToMQ from "@/lib/connectToMQ";
import jwt from "jsonwebtoken";
import { cookies } from "next/headers";

const COOKIE_NAME = 'user-auth';

export async function POST(req, res) {
    try {
        const { id, form } = await req.json();
        console.log(form);
        const token = cookies(req).get(COOKIE_NAME).value;
        const { email } = jwt.verify(token, process.env.JWT_SECRET);
        const channel = await connectToMQ();
        channel.publish('', 'clean-data', Buffer.from(JSON.stringify({
            email, id, form
        })));
        return NextResponse.json({ success: true }, { status: 200 });
    } catch (error) {
        console.error('queue error', error);
        return NextResponse.json({ error: 'internal error' }, { status: 500 });
    }
}