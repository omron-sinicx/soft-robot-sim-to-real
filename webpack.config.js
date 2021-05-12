const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  target: "web",
  entry: {
    index: path.join(__dirname, "src", "js", "index.js"),
    css: path.join(__dirname, "src", "js", "styles.js"),
  },
  output: {
    path: path.join(__dirname, "docs"),
    filename: "[name].bundle.js",
  },
  devServer: {
    hot: true,
    watchContentBase: true,
    liveReload: true,
    port: 8080,
    contentBase: path.join(__dirname, "docs"),
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: "index.html",
      chunks: ["index", "css"],
      template: path.join(__dirname, "src", "html", "index.html"),
    }),
  ],
  module: {
    rules: [
      {
        test: /\.css/i,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.s[ac]ss$/i,
        use: ["style-loader", "css-loader", "sass-loader"],
      },
    ],
  },
};
