using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace DevKen.BitmapExtentionsForOnnx
{
    public static class BitmapExtensionsForOnnx
    {
        public enum TensorType
        {
            Type_13wh, Type_13hw, Type_1hw3, Type_Fast13hw
        }

        public static Tensor<float> ToOnnxTensor(this Bitmap bitmap, TensorType type)
        {
            switch (type)
            {
                case TensorType.Type_13wh:
                    return bitmap.ToOnnxTensor_13wh();
                case TensorType.Type_13hw:
                    return bitmap.ToOnnxTensor_13hw();
                case TensorType.Type_1hw3:
                    return bitmap.ToOnnxTensor_1hw3();
                case TensorType.Type_Fast13hw:
                    return bitmap.FastToOnnxTensor_13hw();
                default:
                    throw new ArgumentException("No mehold to convert to provided TensorType.");
            }
        }

        public static Tensor<float> ToOnnxTensor_13hw(this Bitmap bitmap)
        {
            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Height, bitmap.Width });

            var db = bitmap.SnapShot();

            //bytes [B,G,R,A]
            Parallel.For(0, bitmap.Height, (idx, state) =>
            {
                for (int x = 0; x < db.Width; x++)
                {
                    var pixel = db.GetPixel(x, idx);

                    tensor[0, 0, idx, x] = pixel.B / 255f;
                    tensor[0, 1, idx, x] = pixel.G / 255f;
                    tensor[0, 2, idx, x] = pixel.R / 255f;
                }

            });

            return tensor;

        }

        public static Tensor<float> ToOnnxTensor_13wh(this Bitmap bitmap)
        {

            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Width, bitmap.Height });

            var db = bitmap.SnapShot();

            Parallel.For(0, bitmap.Height, (idx, state) =>
            {
                for (int x = 0; x < db.Width; x++)
                {
                    var pixel = db.GetPixel(x, idx);

                    tensor[0, 0, x, idx] = pixel.B / 255f;
                    tensor[0, 1, x, idx] = pixel.G / 255f;
                    tensor[0, 2, x, idx] = pixel.R / 255f;
                }

            });

            return tensor;
        }

        public static BitmapSnapshot SnapShot(this Bitmap bitmap)
        {
            return new BitmapSnapshot(bitmap);
        }

        public static Tensor<float> ToOnnxTensor_1hw3(this Bitmap bitmap)
        {
            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, bitmap.Height, bitmap.Width, 3 });

            var db = bitmap.SnapShot();

            //bytes [B,G,R,A]
            for (int x = 0; x < bitmap.Width; x++)
            {
                for (int y = 0; y < bitmap.Width; y++)
                {
                    var pixel = db.GetPixel(x, y);

                    tensor[0, y, x, 0] = pixel.B / 255f;
                    tensor[0, y, x, 1] = pixel.G / 255f;
                    tensor[0, y, x, 2] = pixel.R / 255f;
                }
            }

            return tensor;
        }

        //RRRR GGGG BBBB
        public static Tensor<float> FastToOnnxTensor_13hw(this Bitmap source)
        {
            var floatArray = new float[source.Width * source.Height * 3];

            var bitmap_data = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            var bitmap_bytes = new byte[Math.Abs(bitmap_data.Stride) * source.Height];

            Marshal.Copy(bitmap_data.Scan0, bitmap_bytes, 0, bitmap_bytes.Length);

            int total_pixels_count = source.Width * source.Height;


            Parallel.For(0, total_pixels_count, (p_idx, state) =>
            {

                var g_idx = p_idx + total_pixels_count;
                var b_idx = p_idx + total_pixels_count * 2;

                floatArray[p_idx] = bitmap_bytes[p_idx * 3 + 2] / 255f;//R
                floatArray[g_idx] = bitmap_bytes[p_idx * 3 + 1] / 255f;//G
                floatArray[b_idx] = bitmap_bytes[p_idx * 3] / 255f;//B

            });

            source.UnlockBits(bitmap_data);

            return new DenseTensor<float>(new Memory<float>(floatArray), new int[] { 1, 3, source.Height, source.Width });
        }

        public static Bitmap Resize(this Bitmap source, int new_width, int new_height)
        {

            float w_scale = (float)new_width / source.Width;
            float h_scale = (float)new_height / source.Height;

            float min_scale = Math.Min(w_scale, h_scale);

            var nw = (int)(source.Width * min_scale);
            var nh = (int)(source.Height * min_scale);


            var pad_dims_w = (new_width - nw) / 2;
            var pad_dims_h = (new_height - nh) / 2;


            var new_bitmap = new Bitmap(new_width, new_height, PixelFormat.Format24bppRgb);

            using (var g = Graphics.FromImage(new_bitmap))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Low;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;

                g.DrawImage(source, new Rectangle(pad_dims_w, pad_dims_h, nw, nh),
                    0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
            }

            return new_bitmap;
        }


        public static Bitmap ResizeWithoutPadding(this Bitmap source, int new_width, int new_height)
        {

            var new_bitmap = new Bitmap(new_width, new_height, PixelFormat.Format24bppRgb);

            using (var g = Graphics.FromImage(new_bitmap))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Low;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;

                g.DrawImage(source, new Rectangle(0, 0, new_width, new_height),
                    0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
            }

            return new_bitmap;

        }
    }
}
