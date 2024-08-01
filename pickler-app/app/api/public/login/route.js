import connectToDB from "@/lib/connectToDB";
import User from "@/models/UserModel";
import { NextResponse } from "next/server";
import bcrypt from "bcrypt";
import { cookies } from "next/headers";
import jwt from "jsonwebtoken";

export async function POST(req, res) {
    try {
        const { email, password } = await req.json();
        await connectToDB();
        let user = await User.findOne({ email })
        if (!user) {
            const hashedPassword = await bcrypt.hash(password, 10);
            user = await User.create({ email, hashedPassword });
        }
        const token = jwt.sign({ email }, process.env.JWT_SECRET, { expiresIn: '1w' });
        cookies().set('user-auth', token, { maxAge: 60 * 60 * 24 * 7 });
        return NextResponse.json({ user }, { status: 200 });
    } catch (error) {
        console.log('auth error', error);
        return NextResponse.json({ error: 'auth error' }, { status: 500 });
    }
}