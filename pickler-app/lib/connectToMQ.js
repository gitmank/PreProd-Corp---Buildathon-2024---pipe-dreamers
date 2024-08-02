import amqplib from 'amqplib';

const connectToMQ = async () => {
    const conn = await amqplib.connect(process.env.RABBITMQ_URI || 'amqp://localhost');
    const channel = await conn.createChannel();
    return channel;
};

export default connectToMQ;