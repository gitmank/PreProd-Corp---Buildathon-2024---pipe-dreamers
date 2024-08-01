import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import jwt from "jsonwebtoken";

export async function GET(req, res) {
    try {
        const token = cookies(req).get('user-auth').value;
        if (!token) {
            return NextResponse.json({ error: 'auth error' }, { status: 401 });
        }
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        return NextResponse.json({ email: decoded.email }, { status: 200 });
    } catch (error) {
        console.error('auth error', error);
        return NextResponse.json({ error: 'auth error' }, { status: 500 });
    }
}