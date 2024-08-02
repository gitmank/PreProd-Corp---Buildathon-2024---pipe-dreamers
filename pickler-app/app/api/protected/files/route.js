import { NextResponse } from "next/server";
import connectToDB from "@/lib/connectToDB";
import File from "@/models/FileModel";

export async function GET(req, res) {
    try {
        await connectToDB();
        const files = await File.find();
        return NextResponse.json({ files }, { status: 200 });
    } catch (error) {
        console.error('files error', error);
        return NextResponse.json({ error: 'internal error' }, { status: 500 });
    }
}

export async function DELETE(req, res) {
    try {
        const id = new URL(req.url).searchParams.get('id');
        await connectToDB();
        await File.findByIdAndDelete(id);
        return NextResponse.json({ success: true }, { status: 200 });
    } catch {
        return NextResponse.json({ error: 'internal error' }, { status: 500 })
    }
}