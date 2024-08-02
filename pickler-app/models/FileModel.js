import { Schema, model, models } from 'mongoose';

const FileSchema = new Schema({
    name: {
        type: String,
        required: true,
    },
    objectName: {
        type: String,
        required: true,
    },
    owner: {
        type: String,
        required: true,
    },
    status: {
        type: String,
        required: true,
        default: 'uploaded', // uploaded, ready, cleaned
    },
    size: {
        type: Number,
        required: true,
    },
    type: {
        type: String,
        required: true,
    },
    created: {
        type: String,
        default: new Date().toISOString(),
    },
    cleaned: {
        type: String,
        default: '',
    }
});
const File = models.File || model('File', FileSchema);
export default File;