"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.hello = void 0;
const trim_1 = __importDefault(require("./trim"));
exports.default = {
    trim: trim_1.default
};
const hello = () => {
    console.log('Hello.');
};
exports.hello = hello;
